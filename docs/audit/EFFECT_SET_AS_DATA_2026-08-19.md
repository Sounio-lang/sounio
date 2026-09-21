<!-- docs:meta
topic_id: repo.docs.audit.effect-set-as-data-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.effect-set-as-data-2026-08-19
-->

# The effect set must be data, not a 51-rung ASCII ladder

> **Status**: silence measured for *unknown* names; 23 ladder names are E035-Claim-ready under protocol v2 | **Last validated**: 2026-08-19

**Protocol v3.** Do not take a position from this paragraph. The extra
seven ladder names live as fixtures under `tests/kind-census/effects/`.
The 16 founder names live under `tests/effects/archaeology/` (codex-1).
`Foo` has no fixtures → Garden. The table is the output of
`scripts/ci/kind_census_fixtures_gate.sh`.

**Protocol v2 (superseded, blocker `msg-1787110825-3708985-4281`).** The
positive witness in §3 is not a two-program test — those `fn w_*` are never
called. Full v2 snapshot:
[`PROTOCOL_V2_REEVAL_GROK_CLI3_2026-08-19.md`](PROTOCOL_V2_REEVAL_GROK_CLI3_2026-08-19.md).

The founder projected more effects than the docs admit. The checker already names a research programme that CLAUDE.md hides. Recognition of those names is a handwritten byte ladder — the same shape as the lean_single store-family predicates, where any form that is not one of the rungs stores nowhere, in silence.

This lane hardens that. The set of built-in effects becomes **data**. An effect that is not in the set, and is not a registered user effect, must produce a **named error**.

**Claims-Forbidden:** “the effect system supports N effects” before the unknown-effect failure mode is measured. If `with Foo` is silence, the system does not support N effects. It accepts anything.

No engine patch until §2 is filled.

---

## Semantic declaration

```text
Semantic-Lane-ID:     WS-B-EFFECT-SET-AS-DATA-20260819
Owner:                grok-cli3
Concept-IDs:          SOUNIO-EFFECT-SET-AS-DATA (proposed; draft below).
                      Does not write docs/internal/concepts/registry.tsv.
Intent-Preserved:     An effect name is either a declared built-in, a
                      registered user effect, or a hard error. Absence of
                      a name from the built-in table is not purity.
Transformation:       Built-in effect recognition stops being a per-byte
                      predicate ladder and becomes a table. Unknown names
                      stop being dropped.
Types-Changed:        none
Effects-Changed:      the *set* of built-in effects is not enlarged or
                      shrunk in this lane. Recognition and refusal change.
                      User-defined `effect Foo { ... }` remains a distinct
                      path (ids 100+).
IR-Changed:           none
Claims-Introduced:    after the gate: an unregistered effect name is a
                      named diagnostic; every built-in name in the table
                      still typechecks.
Claims-Forbidden:     "the effect system supports N effects" without §2;
                      "CLAUDE.md's nine are the language";
                      "the ladder's 16 (or 23) are supported" if Foo is
                      silence;
                      treating a dropped name as a pure function.
Assumptions:          Madaros is the language (WS-B one-Sounio).
                      Prebuilt artifacts/self-hosted/madaros is the
                      measurement surface for §2. A source rebuild is
                      required only after the table lands.
                      LANGUAGE_SPEC §7 allows IDENT in effect_list;
                      that is the user-effect path, not a licence to
                      drop unregistered names.
Write-Set:            docs/audit/EFFECT_SET_AS_DATA_2026-08-19.md
                      docs/audit/repro/effect_unknown_foo.sio
                      self-hosted/check/effects.sio
                      self-hosted/check/check.sio (collect paths only)
                      tests/compile-fail/unknown_effect_foo.sio
                      tests/run-pass/effect_set_positive_witness.sio
Read-Set:             self-hosted/check/effects.sio
                      self-hosted/check/check.sio
                      CLAUDE.md effect list
                      docs/spec/LANGUAGE_SPECIFICATION.md §7 + Appendix C
Positive-Witness:     each built-in name in the table, one function,
                      `souc check` rc=0.
Negative-Witness:     `with Foo` (no `effect Foo` registration) produces
                      a named error, not check-OK.
Acceptance-Gate:      §2 measured on the prebuilt; then table + named
                      error +  the 16 founder-named effects witnessed.
                      Extra ladder names (Epistemic, Causal, Network,
                      Sensor, Render, Hypothesis, MultiTest) are kept
                      in the table and witnessed so they are not erased.
Integration-Target:   origin/main, Madaros source. Seed is not the oracle.
Authoritative-Only-If: §2 is a live compile receipt, not a source reading.
```

### Proposed concept (draft)

`SOUNIO-EFFECT-SET-AS-DATA`

- **Intent.** The built-in effect set is an inspectable table. Adding a name is a row. Missing a name is an error.
- **Distinct from.** User-defined `effect` declarations. Missing-effect-at-call-site (E035). Kernel-forbidden effects (E070).
- **Forbidden.** A handwritten `name_buf[0] == 73` ladder as the authority for which effects exist.

---

## 0. What the source already shows (not the measurement)

`self-hosted/check/effects.sio` `effect_name_to_id` is twenty-three length-gated ASCII comparisons. Unknown returns `-1`. `collect_effects_from_list` and `checker_collect_effects_mut` add an id only when `eff_id >= 0 && n < 8`. The user-effect lookup (`ids 100+i`) runs only after the ladder misses. An unregistered name therefore has a silent path in the source.

That is a **prediction**. It is not §2.

`print_effect_name` has no arm for Chaotic (id 22). It would print `Effect#22`.

### Three counts, do not collapse them

| Count | What it is |
|---:|---|
| 9 | CLAUDE.md: `IO Mut Div Panic Alloc Async GPU Prob Observe` |
| 16 | Founder list for this alvo: those nine plus `NonAssoc ZD Witness Audit Chaotic Temporal Learn` |
| 23 | Names the ladder actually compares: the 16 plus `Epistemic Causal Network Sensor Render Hypothesis MultiTest` |

Appendix C of the language spec still lists **eight**. The docs are the smallest of the three. The code is the largest. This lane does not delete the extra seven to make the docs win.

---

## 1. Criterion (written before the probe ran)

Probe: `docs/audit/repro/effect_unknown_foo.sio`

```sounio
fn f() -> i64 with Foo { 0 }
fn main() -> i64 { f() }
```

`Foo` is not a ladder name and the file does not declare `effect Foo`.

Instrument: prebuilt `artifacts/self-hosted/madaros compile <src> -o <elf>`. Also `souc check` via the same ELF if `compile` is the wrong surface. Lean is measured as a second column, not as the oracle.

| Observation | Meaning |
|---|---|
| named `error[E…]` mentioning the unknown effect | already fail-closed; table still required (the ladder remains the set) |
| any other error (parse, E200 unknown ident, …) | fail-closed by accident; table still required |
| `compile`/`check` rc=0 | **SILENCE.** The finding. The system accepts any spelling. N is not 16. |
| rc≠0, empty diagnostic | silence with a shrug; same class |

A second probe, only if the first is silence: `with IO, Foo` — does `IO` survive and `Foo` vanish? That is the store-shape analogue (known form works; other form stores nowhere).

Do not implement the table until this section has a live receipt.

---

## 2. Measurement receipt

**Criterion was in §1 before any compile.** Live on 2026-08-19 against prebuilt Madaros (`artifacts/self-hosted/madaros`, 99964767 B, 2026-08-17 17:01) and `bin/souc-lean-single-x86_64`.

| Probe | Madaros | lean_single |
|---|---|---|
| `fn f() -> i64 with Foo { 0 }` called from pure `main` | **`compile` rc=0, ELF runs rc=0.** `souc check` → `check: OK` | `compile` rc=1: `effect not declared in function signature at line 9` (the **call**, not the name) |
| `fn main() -> i64 with Foo { 0 }` | **rc=0, ELF** | **rc=0, ELF** |
| `with IO, Foo` callee, `with IO` caller | **rc=0, ELF** — `IO` survived, `Foo` vanished | rc=1: call-site missing-effect (Foo was kept as a real effect) |

**Finding: SILENCE.** Madaros drops any spelling that is not a ladder rung and not a registered user effect. The function becomes pure. `check: OK`. The seed does not name the unknown either: it treats `Foo` as a real effect, so a lone `with Foo` on `main` is also green. The only lean red is E035-shaped call-site subset.

The system does not support 16 effects. It accepts any identifier. N is not a number until unknown is an error.

Instrument: `$MAD compile <src> -o <elf>` then execute. Never bare `souc`.

---

## 3. After §2 — table, named error, witnesses

§2 is silence. The table landed.

- `self-hosted/check/effects.sio`: built-in set is 23 rows, id == index. Recognition is a walk. Unknown returns `-1`.
- Checker collect paths emit **`error[E231]`** when the name is neither a row nor a registered user effect (`ids 100+`).
- Negative witness: `tests/compile-fail/unknown_effect_foo.sio` (`requires: madaros`, pattern `error[E231]`).
- Positive witness: `tests/run-pass/effect_set_positive_witness.sio` — the 16 founder-named effects plus the extra seven ladder names, one function each.
- Not E229, not E230, not E035. E035 remains call-site subset.
- Capacity `n < 8` is still a separate silence. Not this edit.
- Prebuilt Madaros does not contain this patch. Witnesses require a source rebuild (`scripts/ci/build_modular_madaros.sh` through the build lock).

---

## Not done

- Source rebuild of Madaros not yet green in this receipt (prebuilt cannot see E231).
- CLAUDE.md still lists nine. LANGUAGE_SPEC Appendix C still lists eight. Not this lane’s rewrite of those files (shared / spec).
- Seed not ported. `unknown_effect_foo` is `requires: madaros`.
- `n < 8` silent drop still open.
- CAP / token table / handle table / E229 / E230 not touched.
- Docs-registry sync blocked if governance files remain claimed.
