# Effects Junction — Routing Criterion + Measured Position (fast inliner ⋈ CPS model)

- **Date:** 2026-08-19
- **Author:** fable-1
- **Directive:** founder — deliver the JUNCTION of the compile-time tail-resumptive
  inliner (#1926 fast path) and the `self-hosted/effects/` CPS layer (general path),
  not the choice between them (Koka / Xie & Leijen, ICFP'21).
- **Frame:** Arqueologia v3 — position is COMPUTED by two fixtures (a pass that must
  run; a refuse that must fail with a NAMED diagnostic DIVERGING from the
  unknown-name / `NoSuchType` baseline), never asserted.
- **Headline:** the junction is **Hypothesis**, and the blocker is deeper than
  wiring — the general path **has no execution semantics** and the routing trigger
  is **unparseable**. Both measured below.

## 1. The junction (design)

Two paths, one problem class (`handle`/`perform`/clauses):
- **Fast path** — compile-time inline (#1926). `perform E.op(a)` under an active
  handler lowers to a direct/indirect call of the clause; the clause body value IS
  the perform value. No continuation object, no `resume()`, no CPS. Sound ONLY for
  tail-resumptive clauses.
- **General path** — the `self-hosted/effects/` layer (`efh_perform_effect`, four
  `EfhResumeStrategy`). INTENDED to capture the continuation and support
  Once/Multi/Tail/Never. Measured reality in §3b.

**Default is the general path; the fast path is EARNED.** A clause not *provably*
tail-resumptive falls to the general path — no exception, no warning.

## 2. Routing criterion (operational, for Sounio)

A clause qualifies for the fast path iff **tail-resumptive**: its continuation is
invoked exactly once, in tail position, result returned unchanged — the clause
computes a value and yields it as the perform result, without inspecting, storing,
dropping, or re-invoking the continuation.

| `resume` calls | position | strategy | route |
|---|---|---|---|
| exactly 1 | tail, unchanged | ResumeTail | **fast (inline)** |
| exactly 1 | non-tail / transformed | ResumeOnce | general |
| ≥ 2 | any | ResumeMulti | general |
| 0 | abort/replace | ResumeNever | general |

`clause_is_tail_resumptive(clause)` is a syntactic check (count `resume`, check tail
position). Default-to-general on anything not provably single-tail.

## 3. Measured position — computed, not asserted

> **Measurement integrity — compiler dependency.** The handler-perform bypass is
> in this branch's SOURCE but is UNMERGED, and there is currently no built
> `artifacts/self-hosted/madaros`, so `./bin/souc` silently falls back to the
> committed `bin/madaros-linux-x86_64` (main's compiler, NO bypass). Therefore:
> §3b and §3c are **compiler-independent** (source reading; `resume` parse is
> unchanged) and stand. §3a and §3d are **provisional** — the perform-carrying
> checks below were run against the fallback (main) binary and must be re-measured
> against a build of THIS branch's source before they are load-bearing. Build
> pending (pod build-lock saturated; per founder directive, route via Slurm).

### 3a. Fast path (tail-resumptive) — PASS fixture (PENDING rebased-source build)
`examples/effect_uncertainty_smoke.sio` → `SMOKE 5`;
`examples/effect_uncertainty_gum_vs_mc.sio` → GUM `0.831558` vs Monte-Carlo
`0.851582` (one source, two handlers). The tail-resumptive half executes and
produces real, distinct certified bounds.

### 3b. The general path is a MODEL, not a runtime (verified, handlers.sio)
`efh_perform_effect` (handlers.sio:1216-1225) matches a clause and **records**
`clause.handler_fn_id` into an op-log, then returns `DispatchHandled`:
```
let clause = ctx.clauses[clause_idx as usize]
if ctx.op_count > 0 { ctx.operations[(ctx.op_count-1) as usize].result = clause.handler_fn_id }
...
EfhDispatchResult::DispatchHandled
```
It **never calls `handler_fn_id`, never evaluates a clause body, never yields a
value.** `resume_strategy` is not even read on the perform path.
`efh_compile_handler_clause` (1471-1484) only **increments a synthetic instruction
count** per strategy (`ResumeTail => instr_count + 1  // tail_resume`) and emits no
IR — its own header says "A real implementation would emit actual IR instructions."
`efh_resume` writes a caller-supplied value into `regs[0]`; no continuation runs.
All 65 `test_efh_*` + ~101 `effc_test_*` assert bookkeeping; the closest to
end-to-end (`efh_test_full_pipeline`) never calls its `handler_fn_id=100` — the
resumed value `99` is supplied by the test. Zero importers; `mod.sio` is a
doc-only stub ("not yet integrated ... inference and checking pending").

→ **There is no CPS execution to compare the fast path against.** The equivalence
oracle the directive's deliverable #2 requires does not exist in-tree.

### 3c. The routing trigger (multi-shot) — REFUSE is a GHOST, not a fixture
Measured (`souc check`, no build needed):
- `handle<Choice> { 0 } with { let pick = |k: i64| { k } }` → **`check: OK`**.
- clause with `resume(true); resume(false)` → **`module failed to parse`**.
- bare `resume(1)` → **`module failed to parse`**.

`resume` is a reserved keyword (`lexer/tables.sio:106`, `TokenKind::Resume`) with
**no expression parse rule, no AST node, no checker, no lowering** (`parser.sio:778`
only maps the word to its token kind). A multi-shot handler is rejected today by the
PARSER not knowing `resume` — a generic parse failure **indistinguishable from an
unknown keyword**. By the NoSuchType control this is a **ghost**: the handler
machinery is not discriminating "multi-shot"; name-ignorance is. → the routing
DISCRIMINATION is **Hypothesis** (the refuse side is inexpressible).

### 3d. Unhandled op — coverage is UNCHECKED (PENDING rebased-source build)
The E137 table below was measured against the FALLBACK (main) binary, which has no
bypass at all, so it is main's behavior, not this branch's:
| program (on main binary) | diagnostic |
|---|---|
| `Epistemic.badop` | E137 undeclared + E011 no method |
| `NoSuchEffect.op` | E137 + E011 |
| `frobnicate(1)` | E137 |

**Predicted behavior on THIS branch's compiler** (bypass active,
`check.sio:21137`, `checker_expr_is_handler_perform` = bare-Ident receiver with
`effect_name_to_id >= 0`; `effect_name_to_id("Epistemic") = 8`): the bypass types
ANY `Epistemic.<op>` as `i64` WITHOUT checking a clause exists, so `Epistemic.badop`
**passes check** and fails only at LOWERING (op lookup misses → fallthrough → the
`Epistemic_add` SIGSEGV class). So on this branch coverage is not checked at either
layer with a named diagnostic: check accepts any op; lower fails-open. A real
`DispatchUnhandled` (named, diverging from NoSuchEffect) requires
`check_handler_coverage` (currently callerless) OR a fail-closed lower-time refuse.
**Must be re-measured on the rebased build to confirm the predicted pass-then-SIGSEGV.**

## 4. Equivalence & SOUNIO-VERIFIED-LOWERING
The directive's #2 ("write both, run both, compare") has no oracle: per §3b the
general path produces no value. SOUNIO-VERIFIED-LOWERING (#1955, commit
`e9b4f28e04`) is **not on this branch** (not an ancestor of HEAD), is itself status
**Hypothesis** ("Nothing implements it"), has **no CI gate**, and targets epistemic
e-graph rewrites. So it neither blocks the inliner nor supplies a harness — and even
if it did, §3b means there is no runtime to validate against. The inliner is sound
by the operational argument (a tail-resumptive clause's meaning IS a single tail
call), but cannot carry an empirical translation-validation witness until a real
general path exists.

## 5. Cost & sequence to make the junction real
The blocker is NOT "import `Name` into types.sio" (that only makes the MODEL
typecheck; `types.sio` needs a single `use Name` from `parser/ast.sio`, and
`checker.sio`/`handlers.sio` are self-contained). The blocker is that the executing
runtime the model describes does not exist.
- **J1 — give `resume` a grammar** (parser + `ExprResume` AST + checker + lowering
  where tail-resume ≡ current inline). Prereq for any non-tail handler / refuse-fixture.
- **J2 — routing predicate** in the checker (`clause_is_tail_resumptive`), plus a
  named `DispatchUnhandled` that diverges from E137 (closes §3d's ghost).
- **J3 — build a REAL CPS execution engine** (not wire the model): continuation
  capture + handler-body execution + value-producing resume, emitting actual IR.
  This is a project, not a wire.
- **J4 — equivalence + negative harness** (now possible): a tail handler through
  both real paths, assert equal; a multi-shot handler REJECTED by fast / ACCEPTED by
  general, with the rejection diverging from NoSuchEffect.

Coordination: `self-hosted/check/effects.sio` is grok-cli5's territory (effects, PR
#1963) — untouched here; J1 is pure frontend.

## 6. Claims-Forbidden (what this round does NOT close)
- No real multi-shot (`resume` parse-reserved only).
- No running two-path junction: the general path is a bookkeeping MODEL that never
  executes a handler; there is no CPS execution and no equivalence oracle in-tree.
- Routing is currently TOTAL (100% fast path) — not because everything is proven
  tail-resumptive, but because nothing else is expressible.
- Effect-coverage checking is a GHOST (unhandled op == unknown name diagnostic).
- The inliner carries no SOUNIO-VERIFIED-LOWERING witness (concept absent from
  branch, itself Hypothesis, no gate, no runtime to validate against).
- Per the founder's invitation, this measured refutation — "the junction's general
  arm has no execution and its trigger is unparseable" — is the complete deliverable
  for this round; the design (routing criterion) is ready for when J1–J3 land.
