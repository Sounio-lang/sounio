<!-- docs:meta
topic_id: repo.docs.audit.effect-mod-hold-evidence-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.effect-mod-hold-evidence-2026-08-19
-->

# `with Mod` — evidence for the condition the hold left open

## Why this exists

`self-hosted/check/effects.sio:23-26` records the founder's decision of
2026-08-19:

> `Mod` is HELD, not forgotten. 360 files / 2800 `with Mod` still return -1. **If
> those sites are Mut misspellings, adding Mod would consecrate the error.** No
> reserved hole: a vacant id invites someone to fill it.

The decision is conditional and the condition was never measured. This measures
it.

## Answer

**The evidence supports the misspelling reading.** Three findings, all pointing
the same way, and one that complicates the remedy rather than the diagnosis.

### 1. The functions do what `Mut` describes

`stdlib/safety/imported_runtime_lift_contract.sio:34` and
`stdlib/safety/kernel_replay_evidence_router.sio:30` both open:

    ... -> i64 with Mod {
        var anchor_mask: i64 = 0
        ...
            anchor_mask = anchor_mask + 1

Local mutation, which is exactly the effect `Mut` names. Nothing in these bodies
suggests a distinct "Mod" concept.

### 2. Usage is exclusive, which is what a template produces — not what a typo produces

| | files |
|---|---:|
| use both `with Mod` and `with Mut` | **3** |
| use `with Mod` and never `with Mut` | **360** |

A human typo produces coexistence: the same author writes it correctly
sometimes. Exclusivity across 360 files is the signature of **templated
generation**, where one wrong string is emitted uniformly.

### 3. The distribution confirms a generated family

| directory | files |
|---|---:|
| `tests/run-pass/` | 301 |
| `stdlib/systems/` | 41 |
| `stdlib/theorem/` | 13 |
| `stdlib/safety/` | 5 |
| `tests/effects/` | 2 |
| `self-hosted/check/` | 1 (the hold comment itself) |

The `tests/run-pass/` names are systematic to the point of being mechanical:
`lorenz_i_step_taylor_time_slab_containment_tiny` (×6),
`lorenz_i_step_taylor_local_flowpipe_proof_tiny` (×6),
`solver_portfolio_lorenz_i_step_taylor_response_envelope_v_tiny` (×5).

### 4. What complicates the remedy

**No in-tree generator emits `with Mod`.** The only script mentioning it is
`scripts/archive/sprint160_bootstrap_gate.sh`, which is archived. So the files
are checked in as their own source, and correcting them is a **2,806-site edit
across 363 files**, not one template change. If a generator exists it lives
outside this repository.

## What this does not establish

- That every one of the 2,806 sites means `Mut`. Two samples were read; the
  other 2,804 were counted, not read.
- That the correction is safe. Changing `with Mod` to `with Mut` makes the
  effect *real*, which turns on E035 propagation at every caller of those 2,806
  functions. That propagation has never run. The blast radius is unmeasured and
  is the next thing to measure, before any edit.

## Correction, 2026-08-19 — the counts in this document were wrong

`docs/audit/E035_MOD_BLAST_RADIUS_2026-08-19.md` re-measured and the authoritative
count is **2,813 sites in 365 files**.

Two of my numbers were artefacts, not measurements:

- **2,806** (mine) undercounts by seven: eight sites carry two `Mod` clauses on
  one line and one adjacency was a comment. Arithmetic: 2806 − 1 + 8 = 2813.
- **2,793** was **never a count at all**. The closed-list gate prints twenty
  offending sites and then `omitted=2793`; 2813 − 20 = 2793. I read a **stdout
  cap** as a census and recorded it here as an independent instrument
  disagreeing with mine. It was the same instrument, truncated.

The "13-site disagreement recorded, not averaged" below is therefore also wrong:
there was no second measurement to disagree with. Recording a disagreement was
right; the disagreement was not real.

## Instrument

`git grep -lE '\bwith[[:space:]]+Mod\b' -- '*.sio' ':!archive/*' ':!bootstrap/*'`,
then per-file `grep -qE '\bwith[[:space:]]+Mut\b'` for coexistence. Counts:
363 files, 2,806 sites. Independently, `scripts/ci/effect_name_closed_list_gate.sh`
scanned 165,976 name occurrences and accused 2,846, of which 2,793 are `Mod`.

The two instruments disagree by 13 sites (2,806 vs 2,793) — different exclusion
sets and different clause parsing. Neither is reconciled here; the disagreement
is recorded rather than averaged.

## Claims forbidden

- That `Mod` is confirmed to be `Mut`. The evidence supports it; two files were
  read.
- That the fix is mechanical. Making the effect real turns on propagation that
  has never run.
