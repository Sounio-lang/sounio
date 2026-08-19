<!-- docs:meta
topic_id: repo.docs.audit.kind-ladder-gate-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.kind-ladder-gate-2026-08-19
-->

# Kind-ladder gate — semantic declaration

Written **before** `scripts/ci/kind_ladder_gate.sh`. Protocol v3: the
position is calculated, not asserted. This lane does not change a TypeKind,
an effect, or a registry concept. It changes how a claim about a TypeKind
is allowed to exist.

```
Semantic-Lane-ID: kind-ladder-gate-20260819
Owner: grok-cli5
Concept-IDs: none
Intent-Preserved: a type that only accepts is a label; a refuse that
  starts to pass is a fallen blocker (the known-failure XPASS signal);
  a pass that starts to fail is a regression. Nobody writes a ladder
  position by hand.
Transformation: ladder position becomes the output of running two
  repository fixtures (must-pass, must-refuse with a named diagnostic)
  plus the deepest layer that still names the kind. The table is
  regenerated. It is not an input.
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: the gate discriminates the two bad polarities
  (refuse-passes, pass-fails when not Reserved) because a mutant of
  each was shown to fail the gate this turn.
Claims-Forbidden: "the archaeology is durable" without that positive
  control firing; "all 99 TypeKinds are judged on every PR"; "Garden
  is a hand-written verdict" (Garden is the default for a kind with
  no fixtures); "Executable is a green merge state" (both-pass is
  derived and is also refuse-XPASS — it stops the merge).
Assumptions: Madaros is the semantic engine. lean_single is not an
  oracle. Inherited SOUC_BIN is poison and is unset.
Write-Set: scripts/ci/kind_ladder_gate.sh
  tests/archaeology/kind_ladder/**
  docs/audit/KIND_LADDER_GATE_2026-08-19.md
  docs/audit/TYPE_ARCHAEOLOGY_FAMILY_G_2026-08-19.md
  docs/audit/TYPE_ARCHAEOLOGY_FAMILY_G_2026-08-19.tsv
Read-Set: scripts/lib/gate_assert.sh
  scripts/ci/known_failure_madaros_recheck.sh
  self-hosted/check/types.sio (layer names only)
Positive-Witness: Family G DiffPrivate/DPBudget pass fixtures compile
  to ELF, run rc=0; refuse fixtures emit the named diagnostic.
Negative-Witness: two mutants inside the gate — (1) a refuse fixture
  that is a valid program, (2) a pass fixture that is `i64 = true`.
  If either mutant is accepted, the gate is not a gate.
Acceptance-Gate: bash scripts/ci/kind_ladder_gate.sh
  (selftest mutants fire, then the index is judged)
Integration-Target: origin/main via this worktree's PR
Authoritative-Only-If: the mutant refuse-that-passes fails the gate
  in the same run that prints KIND_LADDER_GATE_OK for the real index.
```

## Day-zero choice

**Judge only kinds that already have fixtures. Grow.**

A TypeKind with no `pass_path` and no `refuse_path` is Garden by default.
The gate does not require 99 pairs. Requiring them on day zero would land
red and be switched off in a week.

Incomplete pair (exactly one path) is Hypothesis. The gate still enforces
polarity on the path that exists: a listed refuse that passes is XPASS; a
listed pass that fails is a regression unless the refuse also fails
(Reserved).

## What stops the merge

Copied from `scripts/ci/known_failure_madaros_recheck.sh`: a tag that
starts passing is a signal, not a silent green.

| polarity | meaning | gate |
|---|---|---|
| listed refuse now type-checks / runs | blocker fell (XPASS) | **fail** |
| listed pass no longer compiles or runs, and refuse did not also fail | regression | **fail** |
| listed refuse fails but the expected diagnostic is absent | wrong refuse | **fail** |
| both fail, refuse names the diagnostic | Reserved | pass (derived) |
| pass runs, refuse fails with the diagnostic | Claim-ready | pass (derived) |
| no fixtures | Garden | pass (not judged) |

Executable (certo passes and refuse also passes) is printed as derived
output and is the XPASS row. It is not a green merge state.

## Cadence (measured this turn)

Engine: this worktree `bin/souc` → `bin/madaros-linux-x86_64` (ELF
`7f454c46`). Inherited `SOUC_BIN` unset.

| run | wall | what |
|---|---:|---|
| `--selftest-only` | 3.496 s | M1 refuse-that-passes + M2 pass-that-fails, both fired |
| `--skip-selftest` (Family G index, 2 judged + 2 Garden) | 2.478 s | ~1.24 s / judged pair |
| full gate (selftest + index) | 6.047 s | default PR invocation today |
| extrapolated 99 judged pairs | ~123 s + 3.5 s selftest ≈ **2.1 min** | well under the full suite |

**Cadence: run on every PR.** Even at 99 pairs this is a short gate, not a
suite. Day-zero index has four rows; it grows as families append fixtures.
Do not require 99 rows to land.

## Positive control (this turn)

A helper that never fails is not a helper. Two mutant indexes were run
through the **process** (not only `judge_kind`):

| mutant | derived | note | gate rc |
|---|---|---|---:|
| refuse = valid `i64` program | Executable | XPASS | **1** |
| pass = `let x: i64 = true` | Hypothesis | REGRESS | **1** |

`KIND_LADDER_GATE_OK` on the real index was printed only after
`KIND_LADDER_GATE_SELFTEST_OK M1=XPASS M2=REGRESS`.

## Semantic-Outcome

```
Semantic-Outcome: positions are now an output of scripts/ci/kind_ladder_gate.sh
Concept-Status-Before: Family G positions were asserted in a markdown census
Concept-Status-After: Family G positions are derived; Fair* Garden by empty paths
Distinctions-Added: Reserved (both fail, named diagnostic) is a green derived state;
  Executable (refuse passes) is derived and is also merge-blocking XPASS
Distinctions-Preserved: monotone ladder; Madaros is the engine; lean_single is not an oracle
Distinctions-Erased: none
Evidence-Run: bash scripts/ci/kind_ladder_gate.sh
  (selftest rc=0; index rc=0; mutant refuse-passes rc=1; mutant pass-fails rc=1)
Fallback-Path: none — without the gate the fixtures rot
Legacy-Kept: scripts/ci/known_failure_madaros_recheck.sh is the pattern, not replaced
Conflicting-Lanes: cursor-1 scripts/dev/typekind_ladder_gate.sh (family A, not this path);
  do not consume other lanes' uncommitted fixtures
Next-Semantic-Interface: families append rows to tests/archaeology/kind_ladder/index.tsv
```
