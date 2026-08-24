<!-- docs:meta
topic_id: repo.docs.audit.effect-enum-2a-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.effect-enum-2a-2026-08-19
-->

# Effect enum 2a — production lookup of the existing 23 ids

Phase 2a of the founder-approved split. This PR is **equivalence only**.
Zero vocabulary decisions. The seven extras belong in 2b, after this
lands, and are not started here.

**Date:** 2026-08-19
**base sha:** `cdea9d7eef` (rebased onto `origin/main` before commit).
`self-hosted/check/effects.sio` on this tip is still the #1956 text
(`e5715c547d`); the two intervening commits are docs/concept only.
**Engine surface:** standalone dual-function test. CI Full Test Suite
compiles it with `/tmp/souc-stage2` (native self-host, lean_single),
not with shipped Madaros. Local `./bin/souc` defaults to Madaros and
is the wrong compiler for that job. The production source change is in
`self-hosted/check/effects.sio`. The shipped Madaros ELF still contains
the old handwritten arms until a rebuilt Madaros is used as the compiler.
**Test:** `tests/run-pass/effect_enum_equiv.sio`

---

## Semantic lane declaration (mandatory)

```text
Semantic-Lane-ID: effect-enum-2a-20260819
Owner: grok-cli5
Concept-IDs: none
Intent-Preserved: the 23 live `with` names keep ids 0-22; unknown stays -1;
  EffVar remains row-polymorphism structure, not a named `with`.
Transformation: representation only. effect_name_to_id consults an enum
  whose declaration order is ID order. No name added, removed, or
  renumbered. No handler, CPS, or inliner choice.
Types-Changed: none
Effects-Changed: none (same 23 ids, same -1)
IR-Changed: none
Claims-Introduced:
  - for each of the 23 names, legacy byte-cmp and enum lookup return the
    same i64
  - for a name in no list (NoSuchEffectX), both return -1
  - EffectKind::EffVar discriminant is 23 and is never returned by lookup
Claims-Forbidden:
  - handlers.sio is connected, or self-hosted/effects/ is on the live path
  - any effect was removed
  - seed FN_EFFECTS bits are production ids (Chaotic is id 22; stdlib
    "Perturbative bit 22" is a bit)
  - extras (Approx, NaturalityG2, Deterministic, Perturbative,
    NarrowWidthApproximation, NonUnitary, Mod) were added
  - Confidence is a new variant (it is an alias of Epistemic)
  - print_effect_name now names Chaotic (id 22 still falls through to
    Effect#22; that string was not part of the equivalence contract)
  - the shipped bin/souc ELF contains the new lookup (it does not, until
    a rebuilt Madaros replaces it)
Assumptions:
  - enum discriminants are sequential from 0 in declaration order
  - user tests cannot call check internals; the dual-function standalone
    file is the id proof
  - SOUC_BIN is poison and is unset
  - any Madaros rebuild is Slurm, never the pod; build_modular_madaros.sh
    is not wrapped in souc-build-lock.sh
Write-Set:
  self-hosted/check/effects.sio
  tests/run-pass/effect_enum_equiv.sio
  docs/audit/EFFECT_ENUM_2A_2026-08-19.md
  docs/audit/EFFECT_ENUM_2A_2026-08-19.tsv
  docs/governance/topic-registry.v1.json
  docs/governance/DOCS_AUTHORITY_MATRIX.md
Read-Set:
  self-hosted/check/effects.sio (pre-change, e5715c547d)
  self-hosted/effects/types.sio (orphan 15-variant enum; not wired)
  /tmp/dispatch_phase2_claude1.md
  /tmp/phase1_effect_enum_union.md
Positive-Witness: NEGATIVE_CONTROL_FIRED old=new=-1
  and EFFECT_ENUM_EQUIV_OK 23/23 + negative, rc=0, BEFORE and AFTER
Negative-Witness: if the negative control does not fire, the measurement
  is null
Acceptance-Gate: BEFORE and AFTER both print NEGATIVE_CONTROL_FIRED and
  EFFECT_ENUM_EQUIV_OK 23/23; souc check self-hosted/compiler/main.sio
  verdict=0 on the new source
Integration-Target: origin/main (own PR; 2b blocked until this lands)
Authoritative-Only-If: the dual-function test is re-run and the negative
  control still fires
```

---

## Semantic outcome

```text
Semantic-Outcome: representation of the existing 23 ids moved from 23
  handwritten byte-cmp arms to an enum in ID order plus a discriminant
  loop. Behaviour of effect_name_to_id is unchanged.
Concept-Status-Before: 23 live names, ids 0-22, unknown -> -1; orphan
  15-variant EffectKind in self-hosted/effects/types.sio disconnected
Concept-Status-After: same 23 ids; production enum in
  self-hosted/check/effects.sio; orphan layer still disconnected
Distinctions-Added: ID order vs historical source-arm order is now the
  enum declaration order (Temporal=20 Learn=21 Chaotic=22)
Distinctions-Preserved: live name vs orphan EffectKind vs seed FN_EFFECTS
  bits vs archaeology pairs; EffVar is structure not a `with` name
Distinctions-Erased: none
Evidence-Run: see BEFORE / AFTER below
Fallback-Path: the frozen legacy_effect_name_to_id in the test file
Legacy-Kept: print_effect_name arms unchanged (Chaotic still Effect#22);
  collect/extract/subset/merge helpers unchanged; self-hosted/effects/**
  untouched
Conflicting-Lanes: grok-cli3 STALE claim on check/effects.sio
  (effect-set-as-data); this lane holds the live claim
Next-Semantic-Interface: phase 2b (seven extras, new ids after 22,
  measure Mod first). Not started.
```

---

## ID table (production, ID order)

| id | name | source-arm order note |
|---:|---|---|
| 0 | IO | |
| 1 | Mut | |
| 2 | Alloc | |
| 3 | Panic | |
| 4 | Div | |
| 5 | GPU | |
| 6 | Async | |
| 7 | Prob | |
| 8 | Epistemic | |
| 9 | Causal | |
| 10 | Network | |
| 11 | Sensor | |
| 12 | Render | |
| 13 | Observe | |
| 14 | NonAssoc | |
| 15 | Audit | |
| 16 | Hypothesis | |
| 17 | MultiTest | |
| 18 | ZD | |
| 19 | Witness | |
| 20 | Temporal | source file listed Chaotic before Temporal/Learn |
| 21 | Learn | |
| 22 | Chaotic | production id 22; not seed bit 22 |
| 23 | *(EffVar)* | not a `with` name; lookup never returns 23 |

---

## Equivalence proof

The test file carries a frozen copy of production `effect_name_to_id` as
of `e5715c547d` and a second function that loops discriminants 0..=22 of
the new enum. User tests cannot call checker internals, so this
standalone dual-function file is the id proof.

If the negative control does not fire, the measurement is null.

### BEFORE (production still handwritten arms)

- when: 2026-08-19T11:00:53Z
- command: `env -u SOUC_BIN -u SOUNIO_SOUC_BIN ./bin/souc run tests/run-pass/effect_enum_equiv.sio`
- host: workspace pod (cheap `souc run` of a standalone file; not a Madaros rebuild)
- output: `NEGATIVE_CONTROL_FIRED old=new=-1` then `EFFECT_ENUM_EQUIV_OK 23/23 + negative`
- rc: 0
- log: session `call-8d24d440-a853-474f-b7c1-f19c853dbf09-108.log`

### AFTER (production source is the enum lookup)

- when: 2026-08-19T11:02:22Z (first) and 2026-08-19T11:06Z (re-run this session)
- same command, same host class
- output: `NEGATIVE_CONTROL_FIRED old=new=-1` then `EFFECT_ENUM_EQUIV_OK 23/23 + negative`
- rc: 0
- log: session `call-878c4197-47b8-492f-b1ea-a22149c10f82-112.log` and this-session re-run

### Production typecheck

`env -u SOUC_BIN -u SOUNIO_SOUC_BIN ./bin/souc check self-hosted/compiler/main.sio`
completed with advisory E-SRB plus truncation/stack warnings and
`verdict=0` / `CHECK_MAIN_RC:0`. The new source typechecks. That is not
a claim that the shipped ELF already contains the new lookup.

### Slurm Madaros rebuild

Required by the dispatch (never on the pod; do not wrap
`build_modular_madaros.sh` in `souc-build-lock.sh`).

- launcher: `scripts/dev/souc-build-remote.sh` (stdin tarball; no local fallback)
- job: 10327
- node: `gpuorangefs-r770-proxmox` (partition `all`, 32 CPUs)
- unpacked: 69M
- `REMOTE: build rc=0 elapsed=226s`
- `REMOTE: elf bytes=100551595`
- the ELF stays on the node; `souc-build-remote.sh` does not ship it back
- this proves the new `check/effects.sio` compiles into Madaros. It is
  not a claim that the workspace `bin/souc` ELF was replaced.

### CI #1960 first run (witness was the wrong measurement)

PR #1960 Full Test Suite (job 96048482036, 2026-08-19T11:15–11:22Z)
reported `FAIL effect_enum_equiv.sio (run exited 1)`. JUnit has no
stdout. That string is the harness wrapping any non-zero `souc run`
(`scripts/dev/run_sio_test_suite.sh` line 402). It is not an assertion
failure.

Measured, cost order:

1. **Compile vs assertion.** `env -u SOUC_BIN ./bin/souc check` (Madaros)
   is `check: OK`. The same file under
   `SOUNIO_SOUC_ENGINE=lean_single` is `typecheck: failed`:
   `tail type mismatch at <main>:11` and
   `effect not declared in function signature` at lines 394/396/402.
   Isolated probes: a bare `[0; 64]` tail of `-> [i8; 64]` is the
   mismatch; `var buf: [i8; 64] = [0; 64]; buf` typechecks. `var`
   assignment in `enum_effect_name_to_id` needs `with Mut` on
   lean_single. Madaros accepts both. The test never reached
   `check_pair` on CI.
2. **Wrong compiler.** Full Test Suite sets
   `SOUNIO_TEST_SOUC_BIN=/tmp/souc-stage2` (artifact of Native Self-Host).
   The PR body measured shipped Madaros. The witness is standalone, so
   the shipped-ELF-lacks-new-lookup claim is not this failure; the
   engine dialect is.
3. **Harness dialect.** `//@ run-pass` plus undeclared Mut / i8-array
   tail. Not an import, not a missing effect on `main`.

The witness was wrong. Equivalence was not disproven. Dialect-only
fix: typed local in `empty_name`, `with Mut` on the enum lookup,
`//@ expect-stdout` for both sentinels. Assertions and ids unchanged.

### AFTER re-run (both engines, 2026-08-19T11:42Z)

| engine | check | run sentinels | rc |
|---|---|---|---|
| Madaros `./bin/souc` | OK | `NEGATIVE_CONTROL_FIRED` + `EFFECT_ENUM_EQUIV_OK 23/23 + negative` | 0 |
| lean_single `SOUNIO_SOUC_ENGINE=lean_single` | compiles (38 fns) | same two lines | 0 |
| harness + `SOUNIO_TEST_SOUC_BIN=bin/souc-lean-single-x86_64` | — | `PASS effect_enum_equiv.sio` | 0 |

21 unexpected passes in the same CI log (including
`gum_fo_across_call.sio`) were not touched.

---

## What this PR does not do

- does not add Approx, NaturalityG2, Deterministic, Perturbative,
  NarrowWidthApproximation, NonUnitary, or Mod
- does not treat Confidence as a new variant
- does not wire `self-hosted/effects/` (handlers.sio stays disconnected)
- does not change `print_effect_name` so Chaotic would print as "Chaotic"
- does not start 2b
- does not merge

Pairs in the exact #1944 pass/refuse layout belong in 2b, after this
lands, and only after Mod is measured (311 `with Mod` files; `-1` today).
