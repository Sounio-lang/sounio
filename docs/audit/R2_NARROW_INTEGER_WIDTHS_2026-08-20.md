<!-- docs:meta
topic_id: repo.docs.audit.r2-narrow-integer-widths-2026-08-20
authority: repo_only
audience: users
last_validated: 2026-08-20
validated_by: lane/minimax-cli2/r2-narrow-integer-widths-2026-08-20
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r2-narrow-integer-widths-2026-08-20
-->

# R2 narrow integer widths - declared widths now mean what they say

**Filed:** 2026-08-20 - **Lane:** minimax-cli2 (worktree at
`/workspace/.wt/minimax-cli2/r2-narrow-int-2026-08-20/`) - **Status:**
semantic declaration filed before any self-hosted change; implementation
deferred to a follow-up commit on this branch.

## Semantic declaration (filed before any code change)

The 6-dimension declaration this work rests on:

- **Concept-IDs.** The founder's R2 dispatch (`/workspace/.wt/claude-1/.scratch/despachos/r2_larguras.md`)
  argues that declared integer widths are part of the meaning of a program.
  The defect is empirical and measured:

  | written | correct in the declared width | printed today |
  |---|---|---|
  | `i8`  `100 + 100` | `-56` | **`200`** |
  | `u8`  `200 + 200` | `144` | **`400`** |
  | `i16` `20000 + 20000` | `-25536` | **`40000`** |
  | `i32` `2000000000 + 2000000000` | `-294967296` | **`4000000000`** |

  Every declared narrow width is `i64` with another name. The existing
  TypeKind enum already names the widths (`TyI8`, `TyU8`, `TyI32`, `TyU32`,
  `TyI64`, `TyU64`, `TyI128`, `TyU128`); the work is to give those names
  semantics in the binary operator and integer literal paths. No new type
  kind is introduced (except where this document calls it out below).

- **Intent-Preserved.** The founder's intent is "each declared width
  truncates / wraps in the width the name promises" - a strict, machine-
  width interpretation; the same rule that C, Rust, and every other
  language with declared integer widths apply. This work preserves the
  intent of the existing narrow-int binop result rule (which already
  follows the left operand's declared type) and adds the missing piece:
  the runtime register must actually mask to that width.

- **Transformation.** Three sites in the lowering chain, all in the HLIR
  layer (no touch of `ir/numeric_payload.sio` or the wide-numeric pool,
  which is R1's lane):

  1. `self-hosted/hlir/ir.sio:727` `hlir_const_int(value, ty)` - mask
     `value` to the bit-width that `ty` carries. Sign-extend for signed
     kinds so `200 as i8 -> -56`, not `200 as u8 -> 200`.
  2. `self-hosted/hlir/lower.sio:2129` `hlir_ast_binary_result_ty_typed`
     - when `lhs_ty` is a narrow integer kind (i8/u8/i16/u16/i32/u32),
     return `lhs_ty` instead of falling through to `hlir_type_i64()`. The
     existing checker-side widening rule (left operand's type drives
     result type) is already correct; the HLIR side was the missing link.
  3. `self-hosted/hlir/builder.sio:197` `hlir_builder_emit_binary` -
     after emitting the i64-register binary op, if the declared result
     type is narrower than i64, emit a `HLIR_OP_TRUNC` (or its IR-level
     equivalent: a const mask + bit-and) that wraps the result into the
     declared width.

  These three together turn `i8 100+100` into the literal `i8 -56` in
  the IR register, with no change to `i64` arithmetic or float
  arithmetic.

- **Claims-Introduced.** (i) The defect is real and the fix surface is
  exactly three sites in `self-hosted/hlir/`; nothing in `ir/`,
  `enir/`, `parser/`, or `check/` needs to move. (ii) `i16` and `u16`
  are NOT in the narrow scope of this work - they currently route to
  `ty_wide_int()` (R1's multi-limb pool, which is explicitly off-limits
  to this lane), and the dispatch acknowledges i16 in the evidence table
  but its fix requires a TypeKind addition (`TyI16`, `TyU16`) that crosses
  into R1's territory. This is a partial scope, documented as such, and
  the PR will say so. (iii) `i128`/`u128` stay in R1's wide-numeric pool
  (they are multi-limb; not a one-register wrap). (iv) Recusa nomeada is
  the form already used in this tree (`E218` for `f128`/`f256`), so a
  new `E219` for `i7`, `i13`, `i999999`, `u3`, `u4096` etc. follows the
  same pattern. (v) A scan of the corpus finds zero real width
  annotations matching `i7`, `u3`, `u4096`, `i999999` etc. (matches are
  local variable names like `u3` for "uncertainty component 3", not
  width annotations), so the recusa has zero compile-fail cost in the
  versioned corpus. The PR measures this and prints the count.

- **Claims-Forbidden.** This PR does NOT introduce a parallel numeric
  mechanism. The narrow-int width truncation is implemented as a mask
  inside the existing i64 register; it does not create a new IR node
  class, does not touch the wide-numeric pool, and does not add a
  parallel binop opcode. This PR does NOT add `TyI16`/`TyU16`; the i16
  /u16 evidence-table rows in the dispatch are deferred, with a `//@
  error-pattern:` test guarding the deferred path so a future lane
  cannot quietly re-enable them via `ty_wide_int` (which silently
  allocates from R1's multi-limb pool). This PR does NOT change `i64`
  /`u64` semantics; both continue to evaluate in a 64-bit register
  without masking, since masking there would be a no-op anyway.

- **Authoritative-Only-If.** **Hypothesis-grade** until the off-pod
  regression run lands and the count of tests that change result is
  measured and reported. The founder's directive is explicit: "corre
  a suite completa e reporta quantos testes mudam de resultado, com os
  nomes. Se algum for de superfície clínica ou da dissertação, para e
  reporta em vez de seguir." The PR is DRAFT until that run lands; no
  merge without the regression count.

## Scope (precise)

In scope (this PR, narrow-int semantics):

- `i8`, `u8`, `i32`, `u32`: TypeKind already exists; HLIR kinds already
  exist; defect is in the lowering, fix is in HLIR. Three sites.
- `i7`, `i13`, `i999999`, `u3`, `u4096` and any other non-power-of-two
  / non-spec width: refused at the parser with `E219` (mirrors `E218`'s
  form for `f128`/`f256`). New diagnostic.

Out of scope (this PR, deferred):

- `i16`, `u16`: TypeKind does not exist; would require adding `TyI16`/
  `TyU16` and the corresponding HlirType kinds, which touches the
  R1-wide-numeric pool indirectly (these widths would still be one-
  register, but they currently route to `ty_wide_int`). Deferred.
- `i128`, `u128`: multi-limb; R1's lane; off-limits.
- `f128`, `f256`: already refused via `E218`; not changed.

The PR will list these scope decisions in the body and the audit doc
will not be updated to claim a wider scope than this.

## Why recusa nomeada (mirroring E218)

The dispatch cites `f128` giving `E218` as the existing form: a source
value type-name that this compiler reserves for itself (compiler-owned
format identity) and refuses to accept in user source. The same form
applies here for `i7`/`i13`/`i999999`/`u3`/`u4096`: the spec at
`docs/spec/LANGUAGE_SPECIFICATION.md` only declares the widths
`i8/i16/i32/i64/i128` and `u8/u16/u32/u64/u128`. Anything else is
either (a) a typo the author can see in the error message, or (b) a
non-spec width the author intends (in which case they should declare
it explicitly when/if the spec grows). The new `E219` says so, with
the same shape as `E218`: parser sets `had_error`, increments
`error_count`, prints the diagnostic with the offending span, and
returns. No checker, IR, SOIR, ABI, or native-lowering path is
reached for the rejected source.

## What changes

(Not yet implemented. This section will be filled in as the
implementation lands; the audit doc is filed before the code.)

| File | Change |
|---|---|
| `self-hosted/hlir/ir.sio` | `hlir_const_int`: mask `value` to the bit-width of `ty`; sign-extend for signed kinds. New helper `hlir_mask_value_to_int_type(value, ty)`. |
| `self-hosted/hlir/lower.sio` | `hlir_ast_binary_result_ty_typed`: return `lhs_ty` when it is a narrow integer kind; no longer fall through to `hlir_type_i64()`. |
| `self-hosted/hlir/builder.sio` | `hlir_builder_emit_binary`: after the binary op, if `ty` is narrower than i64, emit a const mask + bit-and that wraps the result. |
| `self-hosted/parser/types.sio` | New `parser_reject_reserved_narrow_int_path` mirroring `parser_reject_reserved_wide_float_path`, with diagnostic `E219`. Hook into the type-expression parser. |
| `tests/run-pass/os_teus_r2_*.sio` | One positive witness per width (`i8`, `u8`, `i32`, `u32`): `let x: i8 = 100 + 100; print(x);` prints `-56`, etc. Mandatory positive control: a deliberately wrong mask forces the witness to fail. |
| `tests/compile-fail/r2_narrow_int_e219_*.sio` | One compile-fail per refused width (`i7`, `i13`, `i999999`, `u3`, `u4096`); `//@ error-pattern: E219`. |

## Open questions deferred to the founder

- `i16` / `u16` semantics: does the founder want them in narrow scope
  (require `TyI16`/`TyU16` and the corresponding HLIR kinds, touching
  R1's wide-numeric routing) or in R1's wide-numeric pool (one-register
  but allocated from the multi-limb pool)? Dispatch's evidence table
  lists i16 but the lane separation forbids touching R1.
- The dispatch's evidence table shows `i16 20000 + 20000` should print
  `-25536`. If `i16` stays in R1's pool, this PR does not produce that
  result; if the founder wants it, the lane boundary must move.

## FLEET_CONSTRAINTS check

- [x] Work off `origin/main`, on my own branch, my own PR. Worktree at
      `/workspace/.wt/minimax-cli2/r2-narrow-int-2026-08-20/`,
      branch `lane/minimax-cli2/r2-narrow-integer-widths-2026-08-20`,
      rooted at `origin/main`.
- [x] No `git add -A`, no `checkout`, no `stash`, no `clean` in
      `/workspace/sounio/`.
- [x] No full self-compile / `make build` / `lake build` / test suite on
      this pod. Off-pod build via
      `SOUNIO_WITNESS_GLOB='tests/…/os_teus_*.sio' bash scripts/dev/souc-build-remote.sh --gate witness`.
- [x] `./bin/souc` not invoked.
- [x] No Slurm launch.
- [x] PR is DRAFT; I do not merge.
- [x] Atomic commits, one logical change each.
- [x] No `Co-Authored-By` trailer. No AI attribution in commit message.
- [x] EN-UK orthography throughout, including the diagnostic code
      `E219` and all prose.
- [x] Docs registry: `topic-registry.v1.json` synced AFTER the doc
      commit, never before.
- [x] No revert of anyone else's work.
- [x] The narrow-int scope does NOT touch `ir/numeric_payload.sio` or
      the wide-numeric pool (R1's lane).
