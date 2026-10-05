<!-- docs:meta
topic_id: repo.docs.research.ns-wire-n3-handoff-2026-08-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.ns-wire-n3-handoff-2026-08-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# NS wire — N3 (E230 gate) handoff for the build/verify cycle

This is a chronological, historical handoff. Its opening checkpoint predates
the build receipts later in the same document. For the current executable
scope and evidence contract, use
`docs/internal/concepts/ns-antigarbling-lane-20260823.md` and
`scripts/ci/ns_antigarbling_gate.sh`.

At the first checkpoint, implementation was complete but not yet built. The
later `STEP 2` and `STEP 2b` sections preserve what was subsequently measured.

## What N3 added (all in `fable/ns-wire-20260823`, uncommitted on top of the N1+N2 checkpoint)

- `self-hosted/check/check.sio`:
  1. `checker_ns_disable()` — the `SOUNIO_NS_DISABLE=1` sabotage knob (mirrors the
     aleatoric knob), memoized.
  2. `checker_ns_should_refuse(table, bin_op, left_ty, right_ty)` — the E230 predicate:
     independence-assuming op (OpAdd, OpMul) over two Knowledge operands whose source
     sets are not provably disjoint. Empty (deterministic) operand short-circuits to
     "no refusal"; unknown (top) is never disjoint (conservative).
  3. E230 raise at BOTH `knowledge_binary_result` sites (the `*mut Checker` in-place
     path ~6214 and the value path ~21917), so the rule cannot go inert on one.
  4. E230 message (short) + help text.
- `tests/compile-fail/ns_add_shared_source_rejected.sio` — `x + x` (canonical).
- `tests/compile-fail/ns_add_unknown_conservative.sio` — `u + m`, u = top (call return).
- `scripts/ci/ns_antigarbling_gate.sh` — the same-source-built sabotage witness (E230
  refuses, vanishes under `SOUNIO_NS_DISABLE`, an unrelated refusal is unaffected,
  and disjoint Add/Mul are accepted by E230).

## ⚠️ N2 tests that FLIP under N3 — reconcile before the suite goes green

N3 adds a refusal where N2 only emitted a trace. Two existing N2 run-pass witnesses use
operations N3 now (correctly) refuses, so they will fail to compile as written:

1. **`tests/run-pass/ns_unknown_absorb_trace.sio`** — does `u + m` with `u = mystery()`
   (top). N3 refuses this (E230, the conservative case). It is the SAME program as the new
   `ns_add_unknown_conservative.sio` compile-fail fixture. **Reconcile:** delete/convert the
   run-pass witness (superseded), or keep it as a pure N2-dataflow observation by running it
   under `SOUNIO_NS_DISABLE=1` (refusal off, union-absorb trace still emitted).

2. **`tests/run-pass/ns_env_join_overflow_trace.sio`** — builds source-cap overflow via
   repeated `v_i = s + s` self-adds and a final `probe = v64 + v64`. Every `s + s` is a
   shared-source self-add → E230. **Reconcile:** the overflow witness must build its 64+
   sources from DISTINCT measurements (not `s + s`), or run under `SOUNIO_NS_DISABLE=1` to
   observe the pure overflow dataflow.

**Also (build-cycle must do):** audit the ENTIRE run-pass suite — not just `ns_*_trace` — for
any `Knowledge` value added/multiplied with itself or with a top-source operand. Every such
site now emits E230. Expected hits are legitimate anti-garblings (fix the test or the source);
unexpected hits mean the predicate is too aggressive (revisit the empty/disjoint logic).

## Build/verify checklist (step 2)

1. Build Madaros from this worktree (build-lock; heavy build via SLURM per repo policy).
2. `souc check` the two new compile-fail fixtures → expect E230; under
   `SOUNIO_NS_DISABLE=1` → expect E230 to vanish. Do not require `rc=0`, because
   current main may still emit the independent E245 lowering refusal. Confirm the
   `&(*table)` immutable-borrow idiom compiles (first caller of `ns_disjoint` from
   outside the module).
3. Run `scripts/ci/ns_antigarbling_gate.sh` → all controls pass.
4. Run the full run-pass + compile-fail suite; reconcile the two flips above and any suite-wide
   hits; confirm ZERO regressions elsewhere.
5. xai math-review (per §26) → then commit N3 and notify codex on the bus.

## STEP 2 — historical build + verify results (2026-08-25, Madaros v0.80.0 built from these sources)

Built `artifacts/self-hosted/madaros-nswire` (100MB ELF) via `build_modular_madaros.sh`
(serialized through `souc-build-lock.sh`). Runtime verification with that binary:

**PROVEN on that historical lane (the gate worked there):**
- `x + x` (shared source) → `error[E230] ... independence-assuming operation over correlated
  uncertainty`; under `SOUNIO_NS_DISABLE=1` → `check: OK` (sabotage witness).
- `u + m` (u = top via unprojected call) → E230; vanishes under sabotage.
- Disjoint `a + b` and `a * b` (distinct measures) → `check: OK` on that lane
  (not a blanket E230 ban). Current-main integration measures this as “no E230”
  because E245 can still determine the overall command status.
- E222 (R-ORIGIN) under `SOUNIO_NS_DISABLE=1` → **still fires** (knob is NS-specific;
  E230/E222 causally separable).
- `scripts/ci/ns_antigarbling_gate.sh` (SOUC=madaros) → **all controls pass.**

**Blast radius (audit of all 95 Knowledge-arithmetic run-pass tests):** E230=7, OK=77,
OTHER=11. All 7 E230 vanish under sabotage (all NS-caused). All 11 OTHER-ERR persist under
sabotage → **pre-existing failures, NOT caused by this change** (observe_contraction,
seq_*, *_hessian_*, associator_variance_mc, rapamycin_epistemic_adaptive, etc.).

**Of the 7 E230:**
- 5 are the ns_*_trace N2 witnesses (env_join_overflow, unknown_absorb, call_projection_top,
  source_cap_unknown, loop_widen_top) — expected flips, reconcile as below.
- 2 are a **FALSE-POSITIVE CLASS** (must fix before N3 is regression-clean):

## ⚠️ FALSE-POSITIVE CLASS FOUND — deterministic Knowledge defaults to top

`knightian_syntax.sio` and `med/vancomycin_full_propagation.sio` trip E230 on operations
that are actually SOUND. Root cause, isolated by probe:

| Construction form | seeds noise_set_id | verdict |
|---|---|---|
| `measure(...)` | singleton | OK |
| `Knowledge(v, ε=..)` / `Knowledge(v, ε=.., prov=..)` inline ctor | empty | OK |
| **`Knowledge { value: .. }` struct-literal** | **NOT seeded → N1 default -1 (top)** | **false E230** |
| **module-level `let x: Knowledge = ..`** | **NOT seeded → top** | **false E230** (vancomycin `base_dose_per_kg * weight_factor`) |

The N2 empty-seeding covers inline `measure`/ctor but NOT struct-literal construction nor
module-level `let` bindings; those keep the N1 default (top), and top is conservatively
never-disjoint, so their add/mul is wrongly refused. The GATE is correct; the INPUT is
wrongly top.

**Required fix (N2 seeding, before N3 lands green):** seed `noise_set_id = ns_empty()` for
every deterministic Knowledge construction — the `Knowledge { .. }` struct-literal path and
module-level `let` Knowledge bindings — mirroring the inline-ctor seeding. Then rebuild and
re-audit; expected result: knightian_syntax and vancomycin_full_propagation return to OK,
leaving only the 5 ns_*_trace flips to reconcile.

**Reconciliation of the 5 trace flips (after the seeding fix):** run each under
`SOUNIO_NS_DISABLE=1` (N2 dataflow still observable, refusal off) or restructure to distinct
measurements; `ns_unknown_absorb_trace` is superseded by `ns_add_unknown_conservative`.

**NOT committed:** the 100MB `madaros-nswire` binary is a build artifact, not committed.

## STEP 2b — seeding fix + refined diagnosis (2026-08-25, codex-authorized)

codex authorized the N2 seeding fix (seed `ns_empty()` at struct-literal + module-level
Knowledge construction) with four acceptance controls. Implemented and re-diagnosed:

**Struct-literal fix (DONE):** `checker_check_struct_lit_inplace` Knowledge branch
(check.sio ~7408) returned `ty_knowledge(..)` with the N1 default (top). Now seeds
`ns_empty()` + emits a `knowledge_struct_lit` trace, mirroring the inline-ctor path. This
fixes **knightian_syntax** (`Knowledge{..} + Knowledge{..}`).

**Module-level: no separate fix needed.** Probe: a module-level `let base = Knowledge(..)`
(ctor) used in another function's `base * local` → `check: OK`. Module-level `let` preserves
the construction's seed; ctors already seed empty, and the struct-lit fix covers module-level
struct-literals. So "module-level Knowledge construction" is covered by the struct-lit fix;
there is no distinct module-let seeding site.

**vancomycin_full_propagation is NOT a construction-seeding issue — re-diagnosed.** Probe:
a function parameter `w: Knowledge` self-multiplied (`w * w`) with an empty-source argument
still yields E230 → **Knowledge parameters default to top (-1); the caller's argument
source-set is NOT projected into the callee's parameters.** In vancomycin,
`dose2 = dose1 * renal_factor` where both derive from the function's parameters (all top) →
E230. This is the **interprocedural parameter-projection gap** — the §5.6 load-bearing
dependency (the N2 call-projection did the RETURN direction, not parameter seeding), NOT a
construction path. It is OUT OF the requested seeding-fix scope, and a naive "seed params
empty" would be UNSOUND (a caller may pass correlated arguments). Correct fix = interprocedural
argument→parameter source-set projection (future work). Until then, vancomycin is
conservatively refused (sound-but-incomplete), reconciled like the trace flips (run under
`SOUNIO_NS_DISABLE=1`, or defer to the interprocedural work).

## Scope notes (intentional)

- Only **OpAdd** and **OpMul** are gated — the operators §26 names and §2 documents as
  understating. **OpSub** is not gated (its canonical correlated case OVERSTATES = safe);
  **OpDiv** is deferred. A full sign-of-correlation treatment (Sub/Div, negative covariance)
  is future work.
- E230 is causally separate from E245: distinct diagnostic, distinct condition,
  and a dedicated sabotage knob. The current gate proves the separation by
  requiring E230 to vanish while E245 survives on the same source-built compiler.
