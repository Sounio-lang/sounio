<!-- docs:meta
topic_id: repo.docs.audit.unknown-effect-refuse-2026-08-20
authority: repo_only
audience: users
last_validated: 2026-08-20
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.unknown-effect-refuse-2026-08-20
-->

# Unknown effect names must be refused

This declaration precedes the edit. Dispatch 2026-08-20: a made-up
effect name is accepted, and does not participate in the set. The
ninth effect is dropped the same way. Silence is the defect; widening
the `[i64; 8]` table is a different task.

```text
Semantic-Lane-ID: unknown-effect-20260820
Owner: grok-cli2
Concept-IDs: SOUNIO-EFFECT-DECLARATION
Intent-Preserved: with X requires X to be built-in or declared in
  scope; a name the compiler does not know is refused by name; a
  ninth effect that will not fit the eight-slot set is refused by
  name, not dropped
Transformation: unknown closed-list names emit error[E229]; a new
  ninth slot emit error[E232]. Insert/lookup of the 30 recognised
  names, and of user-declared effects, is unchanged.
Types-Changed: none
Effects-Changed: none (surface names, not the effect lattice)
IR-Changed: none
Claims-Introduced: with Zorblex is refused with E229 on the engine
  that was patched; the eight-slot cap is now a named refusal
Claims-Forbidden: raising the slot cap; Mod is Mut; the 2,845
  frozen unknown uses have owners; Madaros is fixed-point-verified;
  this lane edited check.sio while Codex held it
Assumptions: E208 is refinement (taken). E230 is the handle-table
  folklore code; not reused. E231 is grok-cli3's unmerged Foo test.
  E229 (unknown name) and E232 (ninth slot) are free on origin/main
  67aa2aec12. Live Madaros collection is check.sio
  (checker_collect_effects_mut and collect_effects_with_checker),
  not effects.sio extract_effects (no callers).
Write-Set: self-hosted/check/effects.sio,
  self-hosted/compiler/lean_single.sio,
  tests/compile-fail/unknown_effect_zorblex.sio,
  tests/compile-fail/effect_ninth_slot.sio,
  tests/run-pass/effect_known_names_regression.sio, this file
Read-Set: docs/spec/LANGUAGE_SPECIFICATION.md §2.4,
  docs/spec/S06_EFFECTS_ROWS.md §6.1, docs/internal/concepts/effect-declaration.md,
  scripts/ci/effect_name_closed_list_gate.sh
Positive-Witness: souc check of Zorblex currently exits 0 (the hole);
  after the patch it prints error[E229] and exits non-zero
Negative-Witness: with Panic still E035 when missing; with IO still
  checks; user-declared effect Choice still resolves
Acceptance-Gate: compile-fail unknown_effect_zorblex; ninth-slot
  compile-fail under Madaros; known-name regression check: OK
Integration-Target: origin/main 67aa2aec12
Authoritative-Only-If: the pre-patch Zorblex check: OK is recorded,
  and check.sio is patched on the live collect path
```

## Hole, measured 2026-08-20 on origin/main `67aa2aec12`

```
fn g() -> i32 with Zorblex { 0 }
fn main() -> i32 with IO { 0 }
```

| engine | command | result |
|---|---|---|
| Madaros `bin/souc` | `check tests/audit/measure_zorblex.sio` | **check: OK rc=0** |
| lean_single ELF | `souc-lean-single-x86_64 … zorblex.elf` | **ELF 36600 B, rc=0** |
| Madaros control | `with Panic` callee, caller `with IO` | **error[E035] missing Panic, rc=1** |

The line discipline works for names in the table. It does not apply
to a name that is not.

## Two faces of one `&&`

Madaros live path (`check.sio`, Codex holds the file):

```
if eff_id >= 0 && n < 8 { insert }
```

`effects.sio` `collect_effects_from_list` is the same guard and has
**no callers**. Patching only that function would leave Madaros
green on Zorblex.

lean_single: `tok_is_effect_name` is a closed subset. After `with`,
unknown idents are either not consumed (scan) or OR-ed with mask 0
(infer and closures). No diagnostic. The eight-slot array is a
Madaros checker structure; lean_single uses a bitmask, so **E232
does not apply there**.

## Codes

| code | meaning | why not the neighbour |
|---|---|---|
| E229 | unknown effect name | E208 is refinement; next free in the catalogue |
| E232 | ninth effect dropped | E230 is handle-table folklore; E231 is grok-cli3's unmerged Foo |

## Blast radius (not hidden)

A blanket E229 on every name outside the 30-id table also fires on
the frozen accused set (`Mod` ~2261, plus `GUM`, `Foo`, …). Measured
this turn: `with Mod` appears in **59 stdlib files and 306 test
files**. `SOUNIO-EFFECT-DECLARATION` said the refuse may wait on the
tail census. The dispatch still asks for Zorblex to be refused. That
is the same mechanism. Landing it makes those sites red. That is
not a reason to keep Zorblex silent.

## Blocker

`self-hosted/check/check.sio` is held by Codex
(`session-019fcd2c-c730-7391-b120-`, epistemic payload gate).
Coord refused the overlapping claim. Request sent. The Madaros live
hunk is written below and is not applied until that file is free.
No revert, no parallel write.

## Receipts

Pre-patch (the test that had to fail, failing — it was accepted):

```
Madaros  souc check unknown_effect_zorblex.sio  → check: OK rc=0
lean_single seed compile of that file           → ELF 36600 B rc=0
Madaros  with Panic missing                     → error[E035] rc=1
```

lean_single after this lane, compiled on Slurm
`gpuorangefs-multi-r740-proxmox` 2026-08-20T15:50Z, 4 s, ELF 2553281 B:

```
Zorblex  → error[E229]: unknown effect `Zorblex` — name is not in the
           closed list and is not a declared effect
           typecheck: failed  rc=1
known names (Panic, Observe, ZD, IO) → ELF rc=0
```

Madaros live collect is still silent. `check.sio` remains Codex's.
effects.sio `collect_effects_from_list` now reports E229/E232 and has
no callers; that is not the Madaros proof.

```text
Semantic-Outcome: lean_single refuses Zorblex with E229. Madaros does
  not, until checker_collect_effects_mut is patched. The eight-slot
  refusal (E232) is Madaros-only and is not live. The 30 recognised
  names were not removed. Mod remains a frozen accused name; a
  blanket E229 on Madaros will paint those sites red.
```
