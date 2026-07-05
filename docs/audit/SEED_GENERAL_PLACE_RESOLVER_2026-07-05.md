<!-- docs:meta
topic_id: repo.docs.audit.seed-general-place-resolver-2026-07-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.seed-general-place-resolver-2026-07-05
-->

# SEED: general lvalue place resolver + shape-completeness harness — 2026-07-05

Status: **the architectural ("SOTA") fix for the lvalue miscompile class** in the
lean_single bootstrap compiler. Commits `cacd3c358` (resolver + harness) +
`eb45faf5a` (seed resync). Downstream validation: see §4.

## 1. The class, finally named

lean_single compiled lvalue statements and borrows through dozens of
per-token-pattern handler pairs (`stmt_is_<shape>` detector +
`compile_<shape>_x86` emitter). Every shape combination missing from the zoo
was a **silent miscompile**: stores dropped (RHS computed, never written),
temp-copy aliasing (mutations landing on discarded copies), borrows of the
wrong address (slot pointers decoded as data). One family, three eruptions in
three weeks (2026-06-18, 06-22, 07-04→05), each patched point-wise — including
this campaign's own earlier fixes (`cc0122dc0` two-level store materialize,
`c6aba6dbc` borrow tail + ref deref-store).

## 2. The fix — three pillars

1. **`compile_place_projections_x86`** — one recursive place engine walking any
   projection chain (`.field` / `[idx]`) from any root, encoding the
   representation rules exactly once (mirrored from the battle-tested read
   path `compile_postfix_tail`): flat 8-byte field cells; inline nested
   structs/arrays; box/ref cells entered by a single load; aggregate array
   slots holding heap element pointers. Final-place categories (`PLACE_KIND`):
   scalar cell / byte cell / aggregate slot (store = materialize + write ptr,
   borrow = load ptr) / inline aggregate (store = copy_agg, borrow = addr).
2. **General fallback** — `stmt_scan_place_assign` (pure token scan: root
   `name` | `(*name)` + one-or-more projections + `=`; no emission, so no
   backtracking hazards) dispatched **after** all specific handlers, feeding
   `compile_place_assign_general_x86`; and the borrow tail generalized to
   arbitrary continuations through the shared engine. Result:
   **silent store drops are impossible by construction** — a statement either
   matches a proven specific handler, or compiles correctly through the
   general engine, or fails loudly (`tc_error`).
3. **`scripts/ci/lean_lvalue_shape_matrix.{py,sh}`** — a witness generator
   enumerating the shape lattice (roots: local / box / `&!` param / `&` param;
   chains: `.s`, `.inner.a`, `.sarr[k]`, `.aarr[k]`, `.aarr[k].a`,
   `.mid.sarr[k]`, `.mid.aarr[k].a`, `.aarr[k].iarr[j]`; ops: read / assign /
   borrow-mut = 80 coherent combos), compiled+run per combo, gating on 80/80.

Also fixed en passant: the loose duplicate
`stmt_is_deref_field_field_array_store` detector matched without verifying `=`
after the balanced bracket, hijacking `(*p).f1.f2[i].g = v` into the wrong
handler ("unknown identifier" instead of compiling).

## 3. Evidence

| Gate | before | after |
|---|---|---|
| shape matrix | 66/80 (10 borrow COMPILE_ERROR + 4 silent store drops) | **80/80** |
| `lean_borrow_*` witnesses | W1/W2/W4 fixed earlier; W3 via-Box FAIL | **all PASS** |
| two-level RMW witness (W_A) | PASS (kept) | PASS |
| neighbors (aggregate_array_field, nested_ref_field_array_store, rmw_42, nested_struct_field_copy) | PASS | PASS |
| fixed point | — | s3==s4 byte-identical |
| canonical gate | — | PASS (md5 `e774569ca9424334c6dcea0791a4870f`) |

Known remaining defect (separate class, witness committed):
`[[Inner;N];M]` direct 2-D `grid[i][j]` SIGSEGVs at initialization
(`tests/known_failures/lean_field_array_array_aggregate_store.sio`).

## 4. Downstream validation — COMPLETE

Madaros rebuilt from the resynced seed at `eb45faf5a` (clean worktree,
provenance-guarded, 25 IR dumps): **imported matrix 12/12 green** — thin
exit 7, smt witness end-to-end, 6/6 `test_smt_*` ALL PASS, 4/4 `test_dd64_*`
ALL PASS — and the **shape gate 80/80 green (exit 0)** against the deployed
seed. Every gate passes with `ir_patch_validated_calls` live (no skip_patch)
and the finalize/compact passes in their natural two-level RMW form.

## 5. Follow-ups — first two DELIVERED (`be3e63b52` + `be72afbe9`)

- ~~a64 mirror~~ **DONE**: `compile_place_projections_a64` +
  `compile_place_assign_general_a64` (same `PLACE_*` contract; encodings
  mirrored from the proven a64 store handlers) wired into `compile_stmt_a64`.
- ~~Zoo retirement~~ **ROUND 1 DONE**: nine x86 dispatch entries of the
  deref/two-level assign family (incl. a duplicated dispatch pair) + the two
  a64 entries now route through the general engine — the handler family
  behind every silent-store eruption is unreachable. Detector/handler bodies
  remain as dead code for one soak cycle; deletion is round 2. Validated by:
  shape matrix 80/80 end-to-end through the engine, self-compile bounce
  convergence (the compiler compiles itself through the retired shapes),
  canonical PASS (md5 `f45b0296fd6c157776c8c4d3336e49fb`), downstream 12/12.
- Remaining: band-aids inventory (restore_user_main_calls, one-level idioms
  in lower.sio) revertible under the proof protocol; wire
  `lean_lvalue_shape_matrix.sh` into the standard CI gate set; metal-lane
  runtime soak for the a64 engine; zoo round 2 (delete dead handler bodies).
