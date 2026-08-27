<!-- docs:meta
topic_id: repo.docs.audit.effect-archaeology-51-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.effect-archaeology-51-2026-08-19
-->

# Effect archaeology — 51 byte comparisons vs 16 fixtures vs CLAUDE.md 9

Date: 2026-08-19
Lane: `effect-archaeology-51-20260819`
Worktree: `/workspace/.wt/grok-cli3`
Branch: `lane/grok-cli3/effect-set-as-data-20260819` @ 354ad1ce4c (behind `origin/main` 1dc0df549d / #1944)
Instrument: prebuilt `artifacts/self-hosted/madaros` (Madaros v0.80.0, 99964767 B, 2026-08-17 17:01)
Source of the 51 comparisons: `origin/main:self-hosted/check/effects.sio` `effect_name_to_id`
Not used: this worktree's dirty uncommitted rewrite of `self-hosted/check/effects.sio`
Not touched: `self-hosted/effects/` (handlers.sio / checker.sio / types.sio / mod.sio; ~183 KB; never imported; owned elsewhere)

Companion table: [`EFFECT_ARCHAEOLOGY_51_2026-08-19.tsv`](EFFECT_ARCHAEOLOGY_51_2026-08-19.tsv)

---

## Semantic-Lane declaration

```text
Semantic-Lane-ID: effect-archaeology-51-20260819
Owner: grok-cli3
Concept-IDs: none (no registry row names the closed effect set; closest distinction is SEMANTIC_LANE_CONTRACT "effect annotation != physical mechanism")
Intent-Preserved: the operational effect set is what Madaros recognizes, not what a doc list or a fixture directory happens to contain; an unknown spelling is not a reserved name
Transformation: none to compiler semantics. Measurement of effect_name_to_id, plus seven #1944-shaped fixture pairs for names the checker already returns. No new effect, no new diagnostic, no change to E035
Types-Changed: none
Effects-Changed: none (recognition set unchanged; fixtures added for seven already-recognized names)
IR-Changed: none
Claims-Introduced: Madaros recognizes exactly 23 effect identifiers via handwritten byte tests in effect_name_to_id; seven of those lacked archaeology pairs before this lane; those seven are Claim-ready under the two-program test on this instrument
Claims-Forbidden: "supports N effects" while unknown names are silence; "the language has 9 effects" (CLAUDE.md); "the language has 17 effects" (#1944 directory count as cited); "Chaotic is Claim-ready at native" (pass check OK, native emit rc=12); "Foo/Confidence/Deterministic/Approx are reserved" (they are dropped, not refused)
Assumptions: Madaros is the claim-oracle; lean_single may spell more names and has no semantic authority here; prebuilt Madaros implements the same 23-name ladder as origin/main source (confirmed: all 23 produce E035)
Write-Set: tests/effects/archaeology/** (7 new pairs + index rows); docs/audit/EFFECT_ARCHAEOLOGY_51_2026-08-19.md; docs/audit/EFFECT_ARCHAEOLOGY_51_2026-08-19.tsv; docs/audit/repro/effect_foo_{pass,refuse}_shape.sio
Read-Set: origin/main:self-hosted/check/effects.sio; origin/main:tests/effects/archaeology/*; origin/main:scripts/ci/effect_archaeology_gate.sh; CLAUDE.md:245; artifacts/self-hosted/madaros
Positive-Witness: tests/effects/archaeology/{causal,epistemic,hypothesis,multitest,network,render,sensor}_pass.sio (check rc=0 and run rc=0)
Negative-Witness: matching *_refuse.sio (check rc=1, error[E035], named missing effect); docs/audit/repro/effect_foo_refuse_shape.sio (check rc=0, SILENCE — different from E035)
Acceptance-Gate: for each of the 7 new kinds, pass check+run = 0 AND refuse check contains error[E035] AND Foo refuse does not contain error[E035]
Integration-Target: tests/effects/archaeology/index.tsv consumed by scripts/ci/effect_archaeology_gate.sh (file owned by grok-cli5; this lane does not edit the gate)
Authoritative-Only-If: a later Madaros rebuild changes effect_name_to_id; until then the 23 decoded names below are the operational set
```

---

## Measurement 1 — what the 51 comparisons actually test

The census number **51** is the number of *source lines* in `origin/main:self-hosted/check/effects.sio` that contain a `name_buf[i]` test (`rg -c 'name_buf\['` = 51). Several lines pack two or three comparisons. The number of individual `name_buf[i] == N as i8` tests is **132**. Neither 51 nor 132 is a name count.

Decoded from those tests, `effect_name_to_id` returns a non-negative id for exactly **23** distinct identifiers. Unknown input returns `-1`. There is no enum.

| id | name | bytes tested | in #1944 archaeology (before this lane) | in CLAUDE.md:245 |
|---:|---|---:|:---:|:---:|
| 0 | IO | 2 | yes | yes |
| 1 | Mut | 3 | yes | yes |
| 2 | Alloc | 5 | yes | yes |
| 3 | Panic | 5 | yes | yes |
| 4 | Div | 3 | yes | yes |
| 5 | GPU | 3 | yes | yes |
| 6 | Async | 5 | yes | yes |
| 7 | Prob | 4 | yes | yes |
| 8 | Epistemic | 9 | **no** | no |
| 9 | Causal | 6 | **no** | no |
| 10 | Network | 7 | **no** | no |
| 11 | Sensor | 6 | **no** | no |
| 12 | Render | 6 | **no** | no |
| 13 | Observe | 7 | yes | yes |
| 14 | NonAssoc | 8 | yes | no |
| 15 | Audit | 5 | yes | no |
| 16 | Hypothesis | 10 | **no** | no |
| 17 | MultiTest | 9 | **no** | no |
| 18 | ZD | 2 | yes | no |
| 19 | Witness | 7 | yes | no |
| 20 | Temporal | 8 | yes | no |
| 21 | Learn | 5 | yes | no |
| 22 | Chaotic | 7 | yes | no |

Header comment on the same function lists ids 0–21 and omits Chaotic. `print_effect_name` prints names for ids 0–21 and `Effect#N` otherwise, so Chaotic E035 says `missing: Effect#22`. Recognition (id 22) and printing are not the same function.

### Three lists

- **A** (operational, 23): the table above
- **B** (#1944 `tests/effects/archaeology/index.tsv`, 16): Alloc Async Audit Chaotic Div GPU IO Learn Mut NonAssoc Observe Panic Prob Temporal Witness ZD
- **C** (CLAUDE.md:245, 9): IO, Mut, Div, Panic, Alloc, Async, GPU, Prob, Observe

The dispatch cited 17 archaeology effects. The executable #1944 index has **16** kinds (32 `.sio` files + `index.tsv`). Three different lists is the finding.

### Intersection and both-direction differences

```text
A ∩ B ∩ C = C = {Alloc, Async, Div, GPU, IO, Mut, Observe, Panic, Prob}     (9)
A ∩ B     = B                                                                (16)
A ∩ C     = C                                                                (9)
B ∩ C     = C                                                                (9)

A \ B = {Causal, Epistemic, Hypothesis, MultiTest, Network, Render, Sensor}  (7)
A \ C = (A \ B) ∪ {Audit, Chaotic, Learn, NonAssoc, Temporal, Witness, ZD}  (14)
B \ A = ∅
B \ C = {Audit, Chaotic, Learn, NonAssoc, Temporal, Witness, ZD}            (7)
C \ A = ∅
C \ B = ∅
```

Every CLAUDE.md name is recognized and already had a #1944 pair. Every #1944 name is recognized. The holes are only in the other direction: the checker knows seven names the fixture directory did not mention, and fourteen names the programming-guide one-liner does not mention.

---

## Measurement 2 — missing pairs, exact #1944 shape

For each name in `A \ B`, a pair was written by substituting the identifier into the #1944 `io_{pass,refuse}.sio` templates. `diff` of `io_*.sio` with `IO` rewritten to `Epistemic` against `epistemic_*.sio` is empty. Same shape for the other six.

```
fn marked() -> i64 with NAME { 7 }
fn forwarding() -> i64 with NAME { marked() }

fn main() -> i32 with NAME {
    if forwarding() == 7 { 0 } else { 1 }
}
```

```
fn marked() -> i64 with NAME { 7 }
fn drops_effect() -> i64 { marked() }

fn main() -> i32 with NAME {
    if drops_effect() == 7 { 0 } else { 1 }
}
```

Files added (not a new layout):

- `tests/effects/archaeology/causal_{pass,refuse}.sio`
- `tests/effects/archaeology/epistemic_{pass,refuse}.sio`
- `tests/effects/archaeology/hypothesis_{pass,refuse}.sio`
- `tests/effects/archaeology/multitest_{pass,refuse}.sio`
- `tests/effects/archaeology/network_{pass,refuse}.sio`
- `tests/effects/archaeology/render_{pass,refuse}.sio`
- `tests/effects/archaeology/sensor_{pass,refuse}.sio`

`index.tsv` gained seven rows, still five columns (`kind`, `pass_fixture`, `refuse_fixture`, `expected_diagnostic=E035`, `deepest_named_layer=HLIR`). Existing #1944 `IR` marks on Chaotic / GPU / Prob were left alone. The gate script was not edited (grok-cli5 holds it).

### Negative control (mandatory)

A name that does not exist at all, in the same refuse shape:

`docs/audit/repro/effect_foo_refuse_shape.sio` — `with Foo`, `drops_effect` has no effect, `main` declares `Foo`.

| program | check rc | diagnostic |
|---|---:|---|
| any of the 23 `*_refuse.sio` | 1 | `error[E035]` |
| `effect_foo_refuse_shape.sio` | 0 | SILENCE (`check: OK`) |
| `effect_foo_pass_shape.sio` | 0 | SILENCE |

Foo is not E035. The refuse fixtures prove *recognized-and-dropped*, not *unknown-name*. If Foo had produced E035, the fixtures would have been discarded.

Sample recognized refuse (Epistemic):

```
error[E035] in archaeology/epistemic_refuse::drops_effect: effect not declared in function signature (missing: Epistemic) -- required by `marked`
```

Chaotic refuse is still E035, but the printer cannot say the name:

```
error[E035] ... missing: Effect#22 -- required by `marked`
```

That is a print gap, not a recognition gap, and it is still different from Foo silence.

---

## Measurement 3 — one-list-only names on the monotone ladder

Rule used (protocol v2/v3, the correction that re-evaluated nine lanes):

- Garden: no fixtures
- Hypothesis: fixtures exist but the two-program test is not fully satisfied
- Executable: right program passes, wrong program is not refused with the named diagnostic
- Claim-ready: right program passes **and** wrong program fails with the named diagnostic
- Reserva: the name is taken and *every* use is refused with a named diagnostic (both programs fail the same way)

Claim-ready ⇒ Executable ⇒ Hypothesis. Reserva is off the ladder. Both-fail is Reserva, never Claim-ready.

Pass criterion: `madaros check` rc=0 **and** `madaros run` rc=0 (the #1944 gate uses `run` for pass). Refuse criterion: `madaros check` rc≠0 and log contains `error[E035]`.

| name | lists | check pass | run pass | refuse | Foo ≠ refuse? | position |
|---|---|---:|---:|---|---|---|
| Causal | A only (now has pair) | 0 | 0 | E035 | yes | **Claim-ready** |
| Epistemic | A only (now has pair) | 0 | 0 | E035 | yes | **Claim-ready** |
| Hypothesis | A only (now has pair) | 0 | 0 | E035 | yes | **Claim-ready** |
| MultiTest | A only (now has pair) | 0 | 0 | E035 | yes | **Claim-ready** |
| Network | A only (now has pair) | 0 | 0 | E035 | yes | **Claim-ready** |
| Render | A only (now has pair) | 0 | 0 | E035 | yes | **Claim-ready** |
| Sensor | A only (now has pair) | 0 | 0 | E035 | yes | **Claim-ready** |
| Audit | A∩B, not C | 0 | 0 | E035 | yes | **Claim-ready** |
| Learn | A∩B, not C | 0 | 0 | E035 | yes | **Claim-ready** |
| NonAssoc | A∩B, not C | 0 | 0 | E035 | yes | **Claim-ready** |
| Temporal | A∩B, not C | 0 | 0 | E035 | yes | **Claim-ready** |
| Witness | A∩B, not C | 0 | 0 | E035 | yes | **Claim-ready** |
| ZD | A∩B, not C | 0 | 0 | E035 | yes | **Claim-ready** |
| Chaotic | A∩B, not C | 0 | **1** (native write rc=12) | E035 | yes | **Hypothesis** |

Chaotic is the exception that would have been stamped Claim-ready on `check` alone. Isolated `madaros compile chaotic_pass.sio -o /tmp/chaotic_out` ends `Failed to write native binary ... rc=12` / `native-v2 bridge compilation failed`. That is not an effect diagnostic. Under the gate's `run` rule the kind stays Hypothesis. Checker-layer the two-program test looks Claim-ready; native emit does not. `deepest_named_layer` remains the #1944 value `IR`.

The nine names in `A ∩ B ∩ C` were also run on this instrument: all nine are Claim-ready (check 0, run 0, refuse E035, Foo silence). They are not "one-list-only"; they are recorded in the TSV for completeness.

### Not Reserva

No name in A, B, or C is Reserva. Reserva would require the *correct* program to be refused too. Every recognized name accepts a correctly marked + forwarding pair at check.

### Names the founder designed that are not in A

Out of scope to inventory `self-hosted/effects/` (not imported; other lanes). A read-only sweep of `lean_single.sio` effect `src_match` tables, which have no semantic authority, turns up three identifiers **absent from the Madaros 51**:

| name | lean_single | Madaros check of #1944-shaped refuse | position on Madaros |
|---|---|---|---|
| Approx | yes (all three lean tables) | rc=0 SILENCE | Garden (not recognized) |
| Deterministic | yes (`fn_effects`) | rc=0 SILENCE | Garden (not recognized) |
| Confidence | yes (aliased with Epistemic) | rc=0 SILENCE | Garden (not recognized) |

Same diagnostic as Foo. They are not reserved. They are not in archaeology, and this lane did not add pairs for them: a pair for an unrecognized name would fail the negative-control rule (refuse would be silence, identical to Foo).

---

## Commands

```bash
git show origin/main:self-hosted/check/effects.sio | rg -c 'name_buf\['          # 51 lines
git show origin/main:self-hosted/check/effects.sio | rg -o 'name_buf\[' | wc -l # 132 tests
./artifacts/self-hosted/madaros --version
./artifacts/self-hosted/madaros check tests/effects/archaeology/${kind}_pass.sio
./artifacts/self-hosted/madaros run   tests/effects/archaeology/${kind}_pass.sio
./artifacts/self-hosted/madaros check tests/effects/archaeology/${kind}_refuse.sio
./artifacts/self-hosted/madaros check docs/audit/repro/effect_foo_refuse_shape.sio
```

Logs: `/tmp/effect51_measure/`.

No Slurm. No pod compiler rebuild. No edit of `self-hosted/effects/`. No edit of `scripts/ci/effect_archaeology_gate.sh`.

---

## Semantic outcome

```text
Semantic-Outcome: measured the operational effect set at 23 names; closed the 7-name archaeology hole with #1944-shaped pairs; classified one-list-only names; Chaotic held at Hypothesis because native pass fails
Concept-Status-Before: three disagreeing lists (23 / 16 / 9), 7 recognized names without fixtures
Concept-Status-After: same 23-name operational set; archaeology index now has 23 rows; 22 Claim-ready, Chaotic Hypothesis
Distinctions-Added: 51 lines != 23 names; recognized-and-E035 != unknown-and-silent; print_effect_name(Chaotic) != effect_name_to_id("Chaotic")
Distinctions-Preserved: effect annotation != physical mechanism; Madaros != lean_single; Claim-ready != both-fail
Distinctions-Erased: none
Evidence-Run: artifacts/self-hosted/madaros check+run 2026-08-19, receipts under /tmp/effect51_measure/
Fallback-Path: none claimed
Legacy-Kept: #1944 pairs and index columns untouched except seven appended rows; gate script untouched
Conflicting-Lanes: grok-cli5 holds scripts/ci/effect_archaeology_gate.sh and tests/archaeology/** (different tree); fable-1 / cursor-3 hold self-hosted/effects/ (not touched); this worktree still has an uncommitted effect-set-as-data rewrite of self-hosted/check/effects.sio that was not the measurement source
Next-Semantic-Interface: if E231 (unknown-effect named diagnostic) lands, Foo/Approx/Deterministic/Confidence move off Garden and the negative control must be re-run — a same-as-E035 collision would invalidate every refuse fixture
```

Halt. No commit. User merges.
