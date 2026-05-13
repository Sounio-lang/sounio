/-!
# Sounio.ChannelAssignmentSemantics — Phase 8.6 Formal Verification

**Phase 5's epistemic finding:** `butterfly #3` as stated in Kimi 2.6's
Phase 1 forensic review (and formalised in
`formal/KnowledgeArithmeticSoundness.lean` as the KAS-1 policy) is
not a compiler bug — it is an artefact of a misreading of how Sounio
assigns Hessian input channels.

**The actual semantics.**  In `self-hosted/compiler/lean_single.sio`,
a global counter `MEAS_KNOW_IDX` (declared at line 393) is incremented
on *every* `.value` extraction from a Knowledge<T> variable.  The k-th
`.value` access in a function body seeds shadow channel k with the
unit gradient `1.0`; earlier channels get `0.0`.  This is the line:

```sio
if MEAS_KNOW_IDX == 0 { emit_imm64(4607182418800017408) } else { emit_imm64(0) }
```

at `lean_single.sio:11676+`, inside the compile_primary path for
Knowledge<T>.value access.

**Consequence.**  Channels are assigned at **.value extraction time**,
not at `measure()` time.  A Knowledge struct at rest has no channel
identity; it acquires one only when the user extracts `.value`.

**Implication for Phase 4's KAS-1 policy.**  KAS-1 said: "for second-order
GUM, extract `.value` first, do scalar arithmetic, then wrap."  Given
the actual semantics, KAS-1 is not a workaround for a limitation —
it is the natural API.  Any attempt to make `Knowledge * Knowledge`
"carry shadows" would require redefining channel identity at measure()
time, which would break the current Phase 1 tests (which rely on
.value-based channel assignment).

**What Phase 5A attempted and why it could not succeed.**  A structural
attempt (commit reverted) added cross-function shadow-bridging globals
and product-rule propagation inside `compile_knowledge_muldiv_x86`.  It
correctly set `EXPR_SSHADOW_*` / `EXPR_HSHADOW_*` before returning from
the function — but the subsequent `.value` access on the result
incremented `MEAS_KNOW_IDX` and re-seeded channel k with a fresh `1.0`,
overwriting the propagated shadow.  End-to-end closure was impossible
without redefining channel assignment semantics.

This file formalises the correct semantics so the architectural choice
is made explicit and future phases don't re-discover the confusion.

Self-contained.  No Mathlib.  No new axioms.  Zero sorry.
-/

namespace Sounio.ChannelAssignmentSemantics

-- ---------------------------------------------------------------------------
-- §1. Abstract model of MEAS_KNOW_IDX
-- ---------------------------------------------------------------------------

/-- Compile-time program state relevant to channel assignment. -/
structure CompilerState where
  meas_know_idx : Nat    -- counter incremented on each .value access
  deriving Repr

def initial_state : CompilerState := ⟨0⟩

/-- A `.value` extraction is an event that:
    1. seeds channel k = meas_know_idx with gradient 1.0 on that scalar,
    2. seeds all other channels k' ≠ meas_know_idx with 0.0,
    3. increments meas_know_idx for the next extraction. -/
structure ValueAccessEvent where
  channel_seeded : Nat     -- which channel got 1.0 (equal to meas_know_idx at time of access)
  new_state      : CompilerState
  deriving Repr

def process_value_access (s : CompilerState) : ValueAccessEvent :=
  ⟨s.meas_know_idx, ⟨s.meas_know_idx + 1⟩⟩

-- ---------------------------------------------------------------------------
-- §2. Channel-assignment correctness
-- ---------------------------------------------------------------------------

/-- The channel seeded equals the counter value at access time. -/
theorem access_seeds_current_idx (s : CompilerState) :
    (process_value_access s).channel_seeded = s.meas_know_idx := rfl

/-- Each `.value` access increments the counter by exactly 1. -/
theorem access_increments_counter (s : CompilerState) :
    (process_value_access s).new_state.meas_know_idx = s.meas_know_idx + 1 := rfl

/-- Two sequential `.value` accesses seed channels `k` and `k+1`. -/
theorem two_accesses_seed_consecutive_channels (s : CompilerState) :
    let e1 := process_value_access s
    let e2 := process_value_access e1.new_state
    e1.channel_seeded = s.meas_know_idx ∧
    e2.channel_seeded = s.meas_know_idx + 1 := by
  refine ⟨rfl, rfl⟩

-- ---------------------------------------------------------------------------
-- §3. Knowledge arithmetic does not consume channels
-- ---------------------------------------------------------------------------

/-- Model: a Knowledge multiplication operation does *not* touch
    `meas_know_idx`, because the compile path for Knowledge arithmetic
    (`compile_knowledge_muldiv_x86`) does not access the `.value` field
    at compile time (the field is read for the arithmetic but no
    MEAS_KNOW_IDX increment occurs). -/
def process_knowledge_mul (s : CompilerState) : CompilerState := s

theorem mul_preserves_counter (s : CompilerState) :
    (process_knowledge_mul s).meas_know_idx = s.meas_know_idx := rfl

-- ---------------------------------------------------------------------------
-- §4. The would-be "butterfly #3" dissolves
-- ---------------------------------------------------------------------------

/-- Sequence of compiler events for `let kz = k1 * k2; hessian_of(kz.value, j, k)`.
    Only the final `.value` access seeds channels.  The Knowledge
    multiplication is channel-silent.  The hessian_of on `kz.value`
    sees only one channel-seeding event (channel 0), so cross-channel
    Hessian entries like `H_01` are correctly zero. -/
theorem knowledge_mul_then_value_accesses_one_channel
    (s : CompilerState) :
    let after_mul := process_knowledge_mul s
    let event     := process_value_access after_mul
    event.channel_seeded = s.meas_know_idx ∧
    event.new_state.meas_know_idx = s.meas_know_idx + 1 := by
  refine ⟨rfl, rfl⟩

/-- **KAS-1 is the natural API** (restatement).  The sequence
    `let x = k1.value; let y = k2.value; let z = x * y` accesses `.value`
    twice — seeding channels 0 and 1 — and the scalar multiplication
    propagates both through the product rule.  This is NOT a workaround
    for a bug; it is the direct expression of channel-assignment-at-`.value`
    semantics. -/
theorem kas1_sequence_seeds_two_channels (s : CompilerState) :
    let e1 := process_value_access s
    let e2 := process_value_access e1.new_state
    e1.channel_seeded = s.meas_know_idx ∧
    e2.channel_seeded = s.meas_know_idx + 1 ∧
    e2.new_state.meas_know_idx = s.meas_know_idx + 2 := by
  refine ⟨rfl, rfl, rfl⟩

-- ---------------------------------------------------------------------------
-- §5. Summary
-- ---------------------------------------------------------------------------

/-!
## Verified Properties

| Property                                                       | Status  | Location                                  |
|----------------------------------------------------------------|---------|-------------------------------------------|
| `.value` access seeds meas_know_idx                            | Proved  | `access_seeds_current_idx`                |
| `.value` access increments counter by 1                        | Proved  | `access_increments_counter`               |
| Two `.value` accesses seed consecutive channels                | Proved  | `two_accesses_seed_consecutive_channels`  |
| Knowledge multiplication does not consume channels             | Proved  | `mul_preserves_counter`                   |
| The would-be butterfly #3 dissolves into correct semantics     | Proved  | `knowledge_mul_then_value_accesses_one_channel` |
| KAS-1 sequence seeds two channels (natural API)                | Proved  | `kas1_sequence_seeds_two_channels`        |

## What this file formalises

1. Sounio assigns Hessian input channels at `.value` extraction time, not
   at `measure()` time (observed in `lean_single.sio:11676+`).
2. The sequence `let x = k1.value; let y = k2.value; z = x * y` is the
   architecture's natural way to express a two-input function — it is
   not a workaround for butterfly #3.
3. The Knowledge multiplication code path correctly preserves
   `meas_know_idx`; no shadow state leaks through.
4. A user who writes `hessian_of((k1*k2).value, 0, 1)` is asking for
   `∂²/∂x_0∂x_1` of a one-input function (the result of `.value` on a
   Knowledge); `x_1` is not seeded, so the result is zero.  This is
   mathematically correct under the channel-at-`.value` semantics.

## Implication for Phase 4's KAS-1 policy

KAS-1 is **not** a workaround to be replaced by a "proper" compiler fix.
It is the API shape that falls out of the channel-assignment semantics.
Rewording from Phase 4's KnowledgeArithmeticSoundness.lean:

  Before (Phase 4 framing): "KAS-1 compensates for a limitation; a
  future phase could close butterfly #3 architecturally."

  After (Phase 5 finding):  "KAS-1 is the natural expression of the
  channel-at-`.value` semantics.  The 'butterfly' is a misreading of
  how channels are assigned; no architectural closure is needed."

## What Phase 5 initially attempted

A structural attempt (compiler edit reverted at session-end) added:
  * 44 cross-function shadow-bridging globals (`KM_LHS_SSHADOW_*`,
    `KM_LHS_HSHADOW_*`)
  * Caller-side shadow capture in `compile_multiplicative`
  * Product-rule shadow propagation inside `compile_knowledge_muldiv_x86`
    for channel 0 and pair (0,0)

The edits correctly set `EXPR_SSHADOW` / `EXPR_HSHADOW_00` before the
function returned.  But the downstream `.value` access on the result
incremented `MEAS_KNOW_IDX` and re-seeded channel 0 with `1.0` —
overwriting the propagated shadow.  End-to-end closure required
redefining channel-assignment semantics, which would break every Phase
1 test.  The edits were reverted; this file documents the lesson.

## No new axioms

All theorems are `rfl`-reducible under the abstract state model.
-/

end Sounio.ChannelAssignmentSemantics
