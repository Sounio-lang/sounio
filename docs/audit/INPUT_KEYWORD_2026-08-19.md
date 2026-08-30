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

## Addendum 2026-08-27 — rebased onto `main@055825a3f9`; the baseline row moved

This change was written against `6ce6e4dafd`. Between that base and
`main@055825a3f9`, PR #2102 landed `E241` in
`self-hosted/parser/types.sio`: the Knowledge component loop's `else` arm is no
longer a silent skip, it is `report_unknown_knowledge_component`, and a bare
identifier is only an epsilon bound if a comparison operator follows it.

**The premise of this change shifted; it was not refuted.** `Input` still needed
a keyword. What changed is the symptom it removes.

| program | claimed here (base `6ce6e4dafd`) | measured on `main@055825a3f9` **without** this change | measured **with** this change |
|---|---|---|---|
| `Knowledge[f64, Input]` | check rc=0 — "Ident; bracket else-arm skips it" | `error[E241]`, rc=1 — the Ident arm finds no comparison operator and refuses¹ | check rc=0, provenance `AstProvInput` |
| `Knowledge<f64, Input>` | rc=1, `>` eaten as `CmpGt` | unchanged | check rc=0 |

¹ Measured on the sibling identifiers that take the identical path: under the
source-built compiler of this rebased tree, `Knowledge[f64, Source]`,
`Knowledge[f64, Sourc]`, `Knowledge[f64, Literature]` and `Knowledge[f64, 123]`
all return `error[E241]` rc=1, while `Knowledge[f64, Input]` returns rc=0. Before
this change `Input` had no keyword and lexed as `Ident` exactly like `Sourc`.

So the sentence "Not in this change: … the silent `else` skip" is now
**vacuous rather than wrong** — there is no silent skip left to decline. What
this change still does not do is `Source`, `Literature`, the
absent-versus-`derived` collision, `bootstrap_v0.sio`, or `lean_single.sio`.

Adding the keyword does **not** re-open the sink E241 closed: `Input` is now a
`TokenKind`, so it never reaches the Ident-epsilon branch, and every other
unknown component still refuses.

### Re-measured on the rebased tree

Source-built Madaros from this tree (`bash scripts/ci/build_modular_madaros.sh`,
`build_rc=0`, ELF 101433577 bytes — the PR body's 100564677 was the old base's
`main.sio`), local pod, `./bin/madaros check` / `build`:

| program | check rc | run rc | stdout |
|---|---:|---:|---|
| `tests/run-pass/knowledge_provenance_input.sio` | 0 | 0 | `INPUT_KEYWORD_OK` |
| `tests/run-pass/knowledge_provenance_measured.sio` | 0 | 0 | `MEASURED_KEYWORD_OK` |
| `tests/run-pass/knowledge_input_lowercase_ident.sio` | 0 | 7 | (return `7`; intended) |

Gates on the rebased tree:

- `bash scripts/dev/knowledge_annotation_parser_coverage.sh` → **PASS**, after
  re-pointing its pin (see below).
- `bash scripts/ci/knowledge_unknown_component_live_refuse.sh` (source-built
  Madaros) → **pass, total=3 passed=3 failed=0**, positive control fired.

### One gate had to be re-pointed, and that is the rebase's real decision

`scripts/dev/knowledge_annotation_parser_coverage.sh` did not exist at this
change's base. It arrived with E241 and pinned, in three places, that `Input` is
**not** a keyword and `AstProvInput` has **no** parser construction site —
"that would mint a provenance surface we did not ask for". On the rebased tree it
fired three times.

That pin is a tripwire, not a veto: its own header says a moved pin means "either
the gap closed, or the gap widened, and the audit must be re-read". Here the gap
closed, under the founder ruling of 2026-08-19 that this change implements. The
pin was therefore narrowed from `{Source, Literature, Input}` to
`{Source, Literature}`, `constructed` from 3 to 4, and `Input` added to the
must-be-present keyword list so it cannot silently regress.
`docs/audit/KNOWLEDGE_ANNOTATION_PARSER_COVERAGE_2026-08-19.md` carries the
matching addendum.

### One more thing the shipped ELF no longer does

`docs/audit/KNOWLEDGE_ANNOTATION_PARSER_COVERAGE_2026-08-19.md` says of its
DYNAMIC clock that "the committed ELF is expected to still swallow these until
it is rebuilt". Measured 2026-08-27 against `./bin/souc` (Madaros v0.80.0) on
this tree, it does not: `source`, `literature`, `typo_ident`, `int_skip` and
`input` all return rc=1. The shipped ELF has been rebuilt since that sentence
was written.

That is why `input` was removed from the coverage gate's DYNAMIC `fail_probes`
and put in neither list: it refuses under the shipped ELF (which predates this
keyword) and parses under a source-built compiler from this checkout, so any
single expectation there would be false against one of the two clocks.
