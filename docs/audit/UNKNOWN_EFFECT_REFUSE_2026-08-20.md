<!-- docs:meta
topic_id: repo.docs.audit.unknown-effect-refuse-2026-08-20
authority: repo_only
audience: users
last_validated: 2026-08-20
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.unknown-effect-refuse-2026-08-20
-->

# R5 + C1 — Mod joins the closed list; operators require effects; unknown names are refused

This declaration precedes the Madaros live-path edit. Dispatch 2026-08-20:
R5 and C1 land as one change because they fight if split. Admitting `Mod`
is what makes refusing unknown names cheap; requiring `/`→`Div` and `%`→`Mod`
is what makes `Mod` a real effect rather than a silent drop.

```text
Semantic-Lane-ID: unknown-effect-20260820
Owner: grok-cli2
Concept-IDs: SOUNIO-EFFECT-DECLARATION
Intent-Preserved: with X requires X to be built-in or declared in
  scope; a name the compiler does not know is refused by name; a
  ninth effect that will not fit the eight-slot set is refused by
  name, not dropped; operator `/` originates Div and operator `%`
  originates Mod
Transformation: Mod is closed-list id 29 (effect_named_id_max 28→29;
  Confidence remains alias of id 8). Unknown closed-list names emit
  error[E229]. A ninth distinct slot emits error[E232]. `/` without
  Div and `%` without Mod emit error[E233]. Insert/lookup of the
  remaining recognised names, and of user-declared effects, is
  unchanged. Existing undeclared operator sites are not rewritten;
  their count is frozen shrink-only.
Types-Changed: none
Effects-Changed: Mod admitted (id 29). Lattice otherwise unchanged.
IR-Changed: none
Claims-Introduced: with Zorblex is refused with E229; `%` without
  Mod is refused with E233; `/` without Div is refused with E233;
  with Mod / Panic / ZD and the remaining 31 closed names still
  check; the eight-slot cap is a named refusal
Claims-Forbidden: raising the slot cap; Mod is Mut; mass-migrating
  the ~8,057 undeclared operator sites in this change; Madaros is
  fixed-point-verified; Correlated is id 29 (that name is R6, id 30
  after this lane)
Assumptions: E208 is refinement. E230 is handle-table folklore.
  E231 is grok-cli3's unmerged Foo. E229/E232/E233 were free on
  origin/main 67aa2aec12. Live Madaros collection is check.sio
  (checker_collect_effects_mut and collect_effects_with_checker),
  not effects.sio extract_effects (no callers). Seed compile of
  Madaros does not run this checker; `RAW --check main.sio` does.
Write-Set: self-hosted/check/effects.sio,
  self-hosted/check/check.sio,
  self-hosted/compiler/lean_single.sio,
  tests/compile-fail/unknown_effect_zorblex.sio,
  tests/compile-fail/effect_mod_without_decl.sio,
  tests/compile-fail/effect_div_without_decl.sio,
  tests/compile-fail/effect_ninth_slot.sio,
  tests/run-pass/effect_mod_with_decl.sio,
  tests/run-pass/effect_div_with_decl.sio,
  tests/run-pass/effect_known_names_regression.sio,
  scripts/ci/effect_name_closed_list.frozen,
  scripts/ci/operator_effect_ratchet.frozen,
  scripts/ci/operator_effect_ratchet_gate.sh,
  this file
Read-Set: docs/spec/LANGUAGE_SPECIFICATION.md §2.4,
  docs/spec/S06_EFFECTS_ROWS.md §6.1,
  docs/planning/ACTION_PLAN_2026-08-20.md §R5,
  docs/internal/concepts/effect-declaration.md,
  scripts/ci/effect_name_closed_list_gate.sh
Positive-Witness: pre-patch souc check of Zorblex exits 0 (the hole);
  after the patch it prints error[E229] and exits non-zero. `%` and
  `/` without the matching effect were check: OK; after the patch
  they print error[E233]
Negative-Witness: with Panic still E035 when missing; with IO still
  checks; with Mod checks; user-declared effect Choice still
  resolves (it is not in the closed list and remains a declared
  effect, not a silent drop)
Acceptance-Gate: compile-fail unknown_effect_zorblex (E229);
  effect_mod_without_decl / effect_div_without_decl (E233);
  effect_ninth_slot (E232, Madaros); known-name regression
  check: OK; effect_name_closed_list_gate.sh --scan-only;
  operator_effect_ratchet_gate.sh
Integration-Target: origin/main 67aa2aec12
Authoritative-Only-If: the pre-patch Zorblex check: OK is recorded,
  and check.sio is patched on the live collect path, and a rebuilt
  Madaros ELF is the instrument for E229/E232/E233
```

## Why these two changes are one

C1 alone refuses every `with` name outside the closed list. On
`origin/main` that list did not contain `Mod`. `with Mod` appears in
2,814 signatures across 366 files. C1 without R5 paints those red.

R5 without C1 admits `Mod` and then leaves unknown names silent, which
is the hole the closed-list gate is accusing.

Founder ruling 2026-08-20: `Mod` is modular arithmetic, not a `Mut`
typo. It co-occurs with `Mut` in 3 of 2,814 signatures.

## Closed list after this change

`effect_named_id_max()` is 29. Table rows 0–29 plus alias `Confidence`
(id 8) are 31 names. `EffVar` is discriminant 30 and is not a `with`
name.

| id | name |
|---:|---|
| 0–28 | previous production ids (IO … NonUnitary) |
| 29 | Mod |
| 8 (alias) | Confidence → Epistemic |
| 30 | EffVar (not user-facing) |

lean_single is a bitmask, not this id table. `tok_is_effect_name`
accepts `Mod`. `ety_parse_effect_name` returns bit 64 (unused; Observe
is 32, Alloc is 16). That bit is the seed's *declaration* of the name,
not Madaros id 29.

## Operator origin (error[E233])

Neither operator originated an effect. `Div` was written 46,978 times
and gated no checker decision of its own. The natural experiment on
the same tree:

| operator | functions using it | already declaring | compliance |
|---|---:|---:|---:|
| `/` → `Div` | 15,905 | 11,037 | 69% |
| `%` → `Mod` | 3,467 | 278 | 8% |

Live paths: `checker_check_binary_with_operand_types_inplace` and
`Checker.check_binary_with_operand_types`. Const-eval of `/` and `%`
inside the checker is not a user-facing origin and is not E233.

## Residue — measured, not migrated

Dispatch: do not mass-migrate. Freeze the current count; only shrink.

Scan (`scripts/ci/operator_effect_ratchet_gate.sh`), comments/strings
masked, `archive/` and `bootstrap/` skipped:

| region | `/` without Div | `%` without Mod |
|---|---:|---:|
| whole tree (frozen) | 2509 | 1570 |
| `self-hosted/` | 91 | 403 |
| `stdlib/` | 1627 | 352 |
| `tests/run-pass` | 229 fn / 111 files | 137 fn / 116 files |
| `examples/` | 432 | 577 |

The ~8,057 figure in the dispatch is the function-level residue of
*declaring* the effect (4,868 Div + 3,189 Mod by that census). The
ratchet freeze is operator-*site* functions under a stricter slash
heuristic (2509 + 1570). Both numbers are reported; the freeze uses
the gate's own scan so a later tightening of the heuristic cannot
silently raise the bound.

**Halt rule (dispatch):** if mass-migration is required for the
repository to compile, stop and report the number. Seed compile of
Madaros (`build_modular_madaros.sh`) does not run this checker and
still succeeds. `RAW --check self-hosted/compiler/main.sio` under
the new ELF reports **221 error[E233]** (E229=0, E232=0). That is
the number. This lane does not add `with Div` / `with Mod` to those
sites.

## Codes

| code | meaning | why not the neighbour |
|---|---|---|
| E229 | unknown effect name | E208 is refinement |
| E232 | ninth effect dropped | E230 handle-table folklore; E231 grok-cli3 Foo |
| E233 | operator missing its effect | next free after E232 |

## Blast radius (not hidden)

Landing E233 as a hard error means any `souc check` under a Madaros
built from this source will refuse the frozen residue. That is the
mechanism. It is also why this lane does not rewrite those functions.
A suite run against the new ELF will see the 111 `tests/run-pass`
files that use `/` without `Div` go red. That is a follow-on
migration, not this change.

C1 after Mod is admitted does **not** paint the 366 `Mod` files.
The remaining 33 unknown `with` uses (Choice, Foo, Compute, Zorblex, …)
are the closed-list freeze; Zorblex is the compile-fail witness.

## Coordination

`check.sio` was held by Codex (`session-019fcd2c-c730-7391-b120-`).
That claim is STALE. This lane claimed the write set before editing.

grok-cli4 R6 (`Correlated`) copied uncommitted Mod/E229/E233 and
added `Correlated` as id 30 in `/workspace/.wt/grok-cli4` without a
file claim. This lane does **not** add `Correlated`. After R5+C1,
`Correlated` is id 30 and `EffVar` is 31.

## Receipts

Pre-patch (the test that had to fail, failing — it was accepted):

```
Madaros  souc check unknown_effect_zorblex.sio  → check: OK rc=0
lean_single seed compile of that file           → ELF 36600 B rc=0
Madaros  with Panic missing                     → error[E035] rc=1
```

lean_single after commit `9b3c7edfd7`, compiled on Slurm
`gpuorangefs-multi-r740-proxmox` 2026-08-20T15:50Z, 4 s, ELF 2553281 B:

```
Zorblex  → error[E229]: unknown effect `Zorblex` — name is not in the
           closed list and is not a declared effect
           typecheck: failed  rc=1
known names (Panic, Observe, ZD, IO) → ELF rc=0
```

Madaros rebuilt from this source on the same node 2026-08-20T19:43Z,
ELF 100567411 B, `BUILD_MADAROS rc=0 elapsed=546s`. First witness pass
lost diagnostic greps (`rg` absent on the node). Return codes only:

| witness | rc | expected |
|---|---:|---|
| zorblex | 1 | refuse |
| mod_without | 1 | refuse |
| div_without | 1 | refuse |
| known names | 0 | pass |
| mod_with | 0 | pass |
| div_with | 0 | pass |
| NomeQueNaoExiste | 1 | refuse |
| main.sio check | 1 | see E233 count below |

Second rebuild on the same node 2026-08-20T19:55Z, ELF 100567411 B,
`BUILD_MADAROS rc=0 elapsed=546s`, diagnostics via `grep`:

| witness | rc | diagnostic |
|---|---:|---|
| `unknown_effect_zorblex.sio` | 1 | `error[E229]: unknown effect \`Zorblex\` — name is not in the closed list and is not a declared effect` (count 1) |
| `effect_mod_without_decl.sio` | 1 | `error[E233]: operator \`%\` requires \`with Mod\`` (count 1) |
| `effect_div_without_decl.sio` | 1 | `error[E233]: operator \`/\` requires \`with Div\`` (count 1) |
| `effect_ninth_slot.sio` | 1 | `error[E232]: too many effects — the checker set holds 8; the ninth was not recorded` (count 1) |
| `effect_known_names_regression.sio` | 0 | `check: OK` |
| `effect_mod_with_decl.sio` | 0 | `check: OK` |
| `effect_div_with_decl.sio` | 0 | `check: OK` |
| `docs/audit/repro/effect_unknown_name.sio` | 1 | `error[E229]: unknown effect \`NomeQueNaoExiste\` …` (count 2) |
| `self-hosted/compiler/main.sio` | 1 | **E233=221**, E229=0, E232=0. Samples are `%` requires `with Mod`. 58 s |

Pre-patch Zorblex was check: OK. Post-patch it is E229. That is the
positive control.

Local scans, no rebuilt ELF required:

```
EFFECT_NAME_CLOSED_LIST_DERIVED named_id_max=29 table=30 alias=Confidence closed=31
EFFECT_NAME_CLOSED_LIST_SCAN files=7842 hits=166060 known=166027 unknown=33
status=pass  (freeze total=33; Mod removed from the accused set; Zorblex held)
OPERATOR_EFFECT_RATCHET div_without_Div=2509 rem_without_Mod=1570
OPERATOR_EFFECT_RATCHET_OK
```

```text
Semantic-Outcome: Madaros live collect refuses Zorblex (E229), a
  ninth slot (E232), `%` without Mod and `/` without Div (E233).
  Mod is closed-list id 29. Known names including Mod still check.
  Seed self-compile of Madaros succeeded. Checking main.sio with the
  new ELF reports 221 E233; those sites were not rewritten. Residue
  freeze: div_without_Div=2509 rem_without_Mod=1570. Closed-list
  freeze: unknown=33 (Mod removed, Zorblex held).
```
