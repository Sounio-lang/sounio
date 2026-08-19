<!-- docs:meta
topic_id: repo.docs.audit.effect-enum-2b-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.effect-enum-2b-2026-08-19
-->

# Effect enum 2b — Chaotic exists; seven extras after 22

Stacked on phase 2a (`f325084c32`, PR #1960). Founder: Chaotic must exist.

**Date:** 2026-08-19
**base:** `lane/grok-cli5/effect-enum-2a-20260819` @ `f325084c32`
**Mod measured before any extra was added.**

---

## Semantic lane declaration (mandatory)

```text
Semantic-Lane-ID: effect-enum-2b-20260819
Owner: grok-cli5
Concept-IDs: none
Intent-Preserved: ids 0-22 unchanged; unknown stays -1; Confidence is
  Epistemic, not a second thing; seed bits are not production ids.
Transformation: print_effect_name uses the lookup name table so every
  live id has a name (Chaotic=22 prints "Chaotic"). Seven extras receive
  new ids 23-29. Confidence aliases to id 8.
Types-Changed: none
Effects-Changed: seven names that were -1 become ids 23-29. Confidence
  becomes id 8 instead of -1. Chaotic's id is unchanged; only its print
  name is new.
IR-Changed: none
Claims-Introduced:
  - print and lookup are one table; an id with name_len>0 cannot print
    as Effect#N
  - today `with Mod` is silently dropped (360 files, -1 ignored)
  - after this source, Mod is id 29
Claims-Forbidden:
  - handlers.sio is connected, or self-hosted/effects/ is on the live path
  - any effect was removed
  - seed FN_EFFECTS bits were treated as ids (Chaotic is id 22;
    stdlib "Perturbative bit 22" / "NonUnitary bit 22" are BITS)
  - Confidence is a new variant
  - the shipped bin/souc ELF contains this lookup (it does not, until
    a rebuilt Madaros replaces it)
  - adding Mod does not change behaviour (it does: 360 files start
    meaning something)
Assumptions:
  - enum discriminants are sequential from 0
  - SOUC_BIN is poison and is unset
  - Madaros rebuilds go through Slurm
Write-Set:
  self-hosted/check/effects.sio
  tests/run-pass/effect_enum_2b_ids.sio
  tests/run-pass/effect_print_name_cover.sio
  tests/effects/archaeology/{approx,naturalityg2,deterministic,perturbative,nwa,nonunitary,mod}_{pass,refuse}.sio
  tests/effects/archaeology/nosuch_drop.sio
  tests/effects/archaeology/index.tsv
  docs/audit/EFFECT_ENUM_2B_2026-08-19.md
  docs/audit/EFFECT_ENUM_2B_2026-08-19.tsv
  docs/governance/topic-registry.v1.json
  docs/governance/DOCS_AUTHORITY_MATRIX.md
Read-Set:
  tests/run-pass/effect_enum_equiv.sio (unchanged; 2a proof)
  tests/effects/archaeology/io_refuse.sio (#1944 layout)
  self-hosted/compiler/lean_single.sio (seed bits)
  /tmp/dispatch_2b_claude1.md
Positive-Witness: IO drop is E035 (the ruler works); print-cover fails
  if any id in 0..=29 has name_len==0; NoSuchEffectX check is not E035
Negative-Witness: a refuse that matches NoSuchEffectX is not a fixture
Acceptance-Gate: 2a equiv still 23/23 + NEGATIVE_CONTROL_FIRED;
  print-cover OK; 2b ids 23-29 + Confidence=8; Slurm rebuild rc=0
Integration-Target: stacked on #1960, then origin/main
Authoritative-Only-If: print-cover and 2a equiv re-run and still fire
```

---

## 1. Chaotic must exist — print census (before extras)

`print_effect_name` had named arms for ids 0–21 and `else { print("Effect#"); print_int(id) }`.

| id | name | print before 2b |
|---:|---|---|
| 0–21 | IO … Learn | proper name |
| **22** | **Chaotic** | **Effect#22** |
| 23 (then EffVar) | structure | Effect#23 |

Only Chaotic of the 23 live names fell through. That is the list.

Fix: there is no longer a second print table. `print_effect_name` walks
`effect_kind_name_byte`. Generic `Effect#N` only when `name_len(id)==0`
(EffVar, unknown). The test `tests/run-pass/effect_print_name_cover.sio`
fails if any id in 0..=29 has `name_len==0`. That test is the durable
artefact.

---

## 2. Mod measured BEFORE it was added

Instrument: `rg -l --glob '*.sio' 'with Mod'` on this worktree.
Positive control: the same drop shape as #1944 `io_refuse.sio`.

| probe | engine | rc | diagnostic |
|---|---|---:|---|
| `io_drop.sio` (IO, control) | Madaros shipped | 1 | `error[E035] … missing: IO` |
| `mod_drop.sio` | Madaros shipped | **0** | none (`check: OK`) |
| `nosuch_drop.sio` | Madaros shipped | **0** | none (`check: OK`) |
| `mod_decl.sio` | Madaros shipped | 0 | none |

**Finding:** 360 files, 2800 occurrences of `with Mod` (founder said 311;
SHA moved). `effect_name_to_id` returns −1. Both collect paths drop
`eff_id < 0`. The declaration does nothing. Mod and NoSuchEffectX are
the same verdict today. That is worse than absence: 360 files declare
something no compiler reads.

Adding Mod **changes behaviour**. After this source, a drop of Mod is
E035, like IO. The 360 files that call a `with Mod` helper without
declaring it will start to fail check.

---

## 3. The seven extras — new ids after 22

| id | name | seed / comment (a BIT, not this id) |
|---:|---|---|
| 23 | Approx | seed bit 18 = 262144 |
| 24 | NaturalityG2 | seed bit 20 = 1048576 |
| 25 | Deterministic | seed bit 12 = 4096 |
| 26 | Perturbative | stdlib "bit 23" = 8388608 |
| 27 | NarrowWidthApproximation | stdlib "bit 24" = 16777216 |
| 28 | NonUnitary | stdlib "bit 22" = 4194304; **id 22 is Chaotic** |
| 29 | Mod | no seed bit |
| 8 | Confidence | **alias of Epistemic**, same id; not a variant |
| 30 | EffVar | structure; lookup never returns 30 |

Ids 0–22 are unchanged. The 2a test is unmodified and still prints
`EFFECT_ENUM_EQUIV_OK 23/23 + negative`.

---

## 4. Engine divergence (measured, then what 2b closes)

Drop-shape probes on the **shipped** compilers, before this source
existed in any ELF:

| name | Madaros shipped | lean_single seed |
|---|---|---|
| Approx | rc=0 (dropped) | rc=1 effect not declared |
| NaturalityG2 | rc=0 | rc=1 effect not declared |
| Deterministic | rc=0 | rc=0 (name known; does not propagate) |
| Confidence | rc=0 | rc=1 (alias of Epistemic; propagates) |
| Perturbative | rc=0 | rc=1 (unknown → seed catch-all bit 32) |
| NarrowWidthApproximation | rc=0 | rc=1 (bit 32) |
| NonUnitary | rc=0 | rc=1 (bit 32) |
| Mod | rc=0 | rc=1 (bit 32) |

After this source is in a rebuilt Madaros:

- **Closes (name recognised, drop refused):** Approx, NaturalityG2,
  Confidence-as-Epistemic, Perturbative, NWA, NonUnitary, Mod.
- **Stays open:** integer encoding (bits ≠ ids). Deterministic
  *mechanism* (seed: no-IO constraint / E080; Madaros: ordinary
  propagating id 25). Seed still assigns unknown names bit 32 rather
  than these ids. Chaotic still has no seed name.

---

## 5. Proof

- 2a: `tests/run-pass/effect_enum_equiv.sio` — `NEGATIVE_CONTROL_FIRED`
  + `EFFECT_ENUM_EQUIV_OK 23/23 + negative` rc=0 (file not edited).
- Print cover: `EFFECT_PRINT_NAME_COVER_OK 0..=29 named, Chaotic exists`.
- 2b ids: `EFFECT_ENUM_2B_IDS_OK extras 23-29 Confidence=8 EffVar=30`
  + `NEGATIVE_CONTROL_FIRED -1`.
- #1944 pairs under `tests/effects/archaeology/` for the seven extras.
  Negative control `nosuch_drop.sio` must not produce E035. On the
  shipped ELF the extras' refuse fixtures still check-pass (old
  lookup). They become E035 only on a rebuilt Madaros.
- `souc check self-hosted/compiler/main.sio`: 80 E175 ontology, same
  pre-existing set; `self-hosted/check/effects.sio` itself is `check: OK`.

---

## 6. Slurm

- launcher: `scripts/dev/souc-build-remote.sh` (stdin tarball; no local fallback)
- node: `gpuorangefs-r770-proxmox` (partition `all`, 32 CPUs)
- unpacked: 69M
- `REMOTE: build rc=0 elapsed=226s`
- `REMOTE: elf bytes=100561339` (2a was 100551595)
- ELF stays on the node
- this proves the new `check/effects.sio` compiles into Madaros. It is
  not a claim that the workspace `bin/souc` ELF was replaced.
