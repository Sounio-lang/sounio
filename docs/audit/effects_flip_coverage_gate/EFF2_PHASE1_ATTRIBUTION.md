# EFF.2 Phase 1 — attribution of the 66 TYPEFAIL modules (RESULTS)

**Date.** 2026-06-08. **Method.** Deterministic transitive-import-closure split (271-poisoned vs
genuine) + workflow `wf_a17f3cce-f8f` (6 haiku diagnosers + sonnet synthesis) over the 43 genuine,
+ minimal-repro verification of the type-error patterns. Raw: `eff2_phase1_raw.json`.

## Headline: EFF.2's premise is falsified; the per-module gate is the wrong unit

| Category | Count | Owner |
|---|---:|---|
| **271-POISONED** (closure contains a PARSEFAIL/271-wall module) | **23** | the 271-wall (inherited) |
| **GENUINE — REAL_TYPE_ERROR** (genuine, pre-existing checker gaps) | **27** | pre-existing checker bugs (orthogonal to effects) |
| **GENUINE — SELF_NOT_CONTAINED** (bundle-inward types, nothing to seed) | **9** | whole-program check (271-wall-gated) |
| **GENUINE — OTHER** (silent fails / intra-module W035 / 2 false-TYPEFAILs) | **7** | mixed |
| **GENUINE — ALIAS_E008** (the stated EFF.2 motivation) | **0** | — |
| **Total** | **66** | |

(Structured counts; the synthesis agent recounted 9/26/6 — off by one in two buckets; the 9/27/7/0
above are authoritative from the per-module JSON.)

## The four findings

### 1. ALIAS_E008 = 0 — EFF.2's premise does not occur

EFF.2 was motivated by imported type **aliases** not being seeded (`E008`). **Zero of the 43 genuine
modules fail this way.** The alias gap is real in a synthetic probe but absent from the actual
compiler surface. **Proceeding with EFF.2's per-module seeding work would fix zero confirmed root
causes.**

### 2. SELF_NOT_CONTAINED = 9 — the per-module gate is structurally unsound

`check/{borrow,compat,effects,env,epistemic,hyper,refinement,units}.sio`, `hlir/opt_strategy.sio`
have **zero `use` imports** yet reference sibling types (`Name`, `TypeEntry`, `Checker`, `FnSig`, …)
that flow **inward from the bundle root** (`main.sio`/`check.sio`), not upward from the leaf. A
per-module check cannot resolve them and there is **nothing to seed**. Checking these in isolation is
the wrong unit — neither pass nor fail is meaningful. The correct unit is the **whole-program
(bundle) check**, blocked by the 271-wall.

### 3. REAL_TYPE_ERROR = 27 — genuine, pre-existing, orthogonal to effects

These are real checker gaps (verified by minimal no-import repros — they are *not* per-module
artefacts), in two systematic patterns plus a tail:

- **Array-literal element-coercion (~14):** `var buf: [i8; N] = [0; N]` — the fill literal `[0; N]`
  is typed `[i64; N]`, not the declared element type → `E001`/`E016`. (Minimal repro `[i8;4]=[0;4]`
  → E001.) Extends the known scalar literal-coercion gap to array initialisers. Hits `gpu/*`,
  `io/*`, `interop/protocol`, `native/macho`.
- **Uninitialised-struct inference (~6):** `var s: S` then field-assign infers `s : ()` not `S` →
  `E001`. (Minimal repro confirms.) Hits `ir/egraph`, `native/{elf,encode,frame,reloc}`.
- **Tail (~7):** `as usize` casts (`usize` not a Sounio type) + misc `E004`/`E005`/`E007`.

These are **type** bugs, **not** effect-enforcement findings. They block per-module checking of
those modules; in the bundle the compiler currently builds (via `bin/souc`), so they may be checker
gaps the bundle path tolerates — but they reproduce minimally, so they are genuine.

### 4. OTHER = 7 — incl. 2 false-TYPEFAILs

`check/{dependent,lint,ownership}.sio` fail silently (no error code — diagnostic gap);
`check/layout.sio` (W035 Div) and `gpu/lower_to_ptx.sio` (E035) are intra-module effect warnings;
**`interop/{serialize,server}.sio` actually pass** in isolation → mis-classified as TYPEFAIL by the
Phase-B harness (audit upstream).

## Consequence — retire the per-module approach

**Phase 1 undercuts both EFF.1's per-module coverage gate (Fb) and EFF.2's seeding premise:**

- EFF.2 (ALIAS_E008 seeding): **0 occurrences → no longer justified.** Do not implement.
- EFF.1 per-module gate: **structurally unsound** for the 9 SELF_NOT_CONTAINED modules (and noise-
  dominated by 27 unrelated type bugs + 23 271-poisoned). The right unit is the **whole-program
  check**.

**The effects-flip coverage therefore reduces to a single gating dependency plus orthogonal cleanup:**

1. **Fix the 271-wall** (keyword demotion; concurrent session). This (a) clears the 23 poisoned, and
   (b) restores the whole-program check, which is the correct unit for the 9 SELF_NOT_CONTAINED and
   for the whole compiler. Then run the existing whole-program check under `toggle = 2` — that
   enforces effects across every module body in one pass. **No per-module gate, no seeding work.**
2. **(Orthogonal, optional)** fix the two systematic checker type-gaps (array-literal element
   coercion; uninit-struct inference) — pre-existing, well-characterised, independent of effects.

## Caveats

- Cause labels are heuristic (single-pass agent reading compiler output); 3 silent failures
  (`dependent`/`lint`/`ownership`) are unattributed beyond "OTHER".
- Warn-mode: `layout`/`lower_to_ptx` effect warnings would become hard errors under `toggle = 2`.
- "0 ALIAS_E008" is over *these 43 modules at warn-mode*; not a global proof the alias gap can never
  bite — but it is absent from the real coverage surface, which is what matters for prioritisation.
- The REAL_TYPE_ERROR "genuine" claim is verified by minimal repro; whether the *bundle* path
  tolerates them (it builds today) is a separate question gated by the 271-wall.
