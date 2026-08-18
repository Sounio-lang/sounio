<!-- docs:meta
topic_id: repo.docs.audit.e137-two-causes-diagnosis-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.e137-two-causes-diagnosis-2026-08-18
-->

> **Status**: Production | **Last validated**: 2026-08-18 | **Source**: `bin/souc check` reproductions, `self-hosted/check/check.sio:19734-19746`

# E137 in the 12 toolchain defects — two distinct causes sharing one error code

**Date:** 2026-08-18
**Scope:** E137 "use of undeclared variable" in the two test files flagged in today's dissertation triage: `examples/dissertation_scenario_gate_demo.sio` (with `print_i64` in `bbb_voi`) and `tests/run-pass/rapamycin_kaxi_fuse_prior.sio` (with `acknowledge` in `main` and in `kaxi::kaxi_fuse`)
**Verdict:** **Two distinct root causes**, same E137 surface, same typecheck dispatch code path (`check.sio:19734`). Different fixes on different lanes.
**Lane:** minimax-cli3 (this lane); the fixes live in grok-cli2's stdlib and in self-hosted/check's typechecker.

## Why this doc exists

Today is the third time a single code has hidden more than one defect. The fleet now has a pattern worth writing down:

| Defect surface | Disposition | What it hid |
|---|---|---|
| `lower_array` | two causes | different bugs surfaced as the same symptom |
| `rc=182` | one cause | the whole family was one root, after four independent measurements |
| **`E137`** (this doc) | **two causes** | print_i64 visibility ≠ acknowledge typecheck registration |

Each case looked the same on first reading. Each needed a measurement before the right fix could be assigned. The "is this one cause or two" question is cheap to ask and expensive to answer wrong — the cost of getting it wrong is the same as the rc=182 fleet mistake of citing a table for two months, applied to which lane writes which fix.

## What I measured

Compiled both files against the prebuilt `bin/souc` (per FLEET_CONSTRAINTS — `./bin/souc` is prebuilt, used as-is, no rebuild attempted).

`examples/dissertation_scenario_gate_demo.sio`:

```
error[E137] in bbb/bbb_voi::bbb_voi_print at 4885..4894: use of undeclared variable
   |
   = help: declare the variable before use, or import it from another module
   = name print_i64
```

`tests/run-pass/rapamycin_kaxi_fuse_prior.sio`:

```
error[E137] in run-pass/rapamycin_kaxi_fuse_prior::main at 1856..1863: use of undeclared variable
   = name seq_new
error[E137] in run-pass/rapamycin_kaxi_fuse_prior::main at 2340..2351: use of undeclared variable
   = name acknowledge
error[E137] in epistemic/kaxi::kaxi_fuse at 1013..1024: use of undeclared variable
   = name acknowledge
```

The error names are different. The dispatch is the same. The byte offsets differ. Three call sites, three distinct callees, two distinct root causes.

## Cause A — `print_i64` (visibility / missing `pub`)

`print_i64` **exists** in the stdlib, twice:

- `stdlib/plot/bar.sio:301` — `fn print_i64(value: i64) with IO` *(not `pub`)*
- `stdlib/metrology/calibration.sio:322` — `fn print_i64(x: i64) with IO, Mut, Panic` *(not `pub`)*

Neither is `pub`. Neither is re-exported from a commonly-imported module. `bbb_voi.sio` imports only `bbb_hdmr::{BBBHDMRResult}` and `bbb_priors::{BBBPriors}` — neither pulls `print_i64` into scope.

The typechecker at `check.sio:19734` looks the name up in `env` (miss — `print_i64` is not local to `bbb_voi_print`) then in `fn_sigs` (miss — `print_i64` has no entry there), and emits E137.

**This is a system-wide visibility gap**, not a per-file bug. The same E137 fires at every call site:

| File | Byte offset | Source line |
|---|---|---|
| `bbb/bbb_voi::bbb_voi_print` | 4885..4894 | `print_i64(rank_k)` |
| `bbb/bbb_gate::bbb_gate_print_verdict` | 4579..4588 | `print_i64(n)` |
| `pd/pd_gate::pd_gate_print` | 5397..5406, 5534..5543, 5590..5599 | three call sites |
| `pd/pd_gum::pd_voi_print` | 12282..12291 | `print_i64(rank_k)` |
| `scenarios/steady_state_runner.sio::print` | multiple | three call sites |

`fn print_i64` was written but never `pub`. Every call site in the stdlib fails the same way.

**The defect is in the stdlib source.** The fix lives on grok-cli2's lane:

- Add `pub` to both definitions (`stdlib/plot/bar.sio:301`, `stdlib/metrology/calibration.sio:322`).
- Add a re-export from a commonly-imported module (likely `stdlib/io/mod.sio` or whatever the print-helper convention lives under) so any darwin_pbpk file can `use stdlib::io::*` and pick it up.
- Alternative (smaller fix): just add `pub` to the existing definitions and add `use darwin_pbpk::metrology::calibration::{print_i64}` to the four call sites that lack it. This keeps the fix local but is uglier.

## Cause B — `acknowledge`, `seq_new`, `observe` (codegen intrinsic, no typecheck stub)

`acknowledge` does **not exist** as a function anywhere in source:

```
$ grep -rn "fn acknowledge\b" --include="*.sio" stdlib/ /workspace/.wt/grok-cli2/stdlib/
(no matches)
```

The closest match is `acknowledge_d10_site_a_handoff` — a different name, in `proof_carrying_deployment_validity_revocable_authority.sio`. Nothing called `acknowledge` is defined.

What exists is **codegen-time intrinsic handling**:

- `self-hosted/compiler/lean_single.sio:10249` — `fn compile_acknowledge_call_x86(call_tok)`
- `self-hosted/compiler/lean_single.sio:31693` — `fn compile_acknowledge_call_a64(call_tok)`
- Dispatched at `lean_single.sio:13022` (x86) and `lean_single.sio:33505` (a64) when `fn_find` returns negative — i.e., when no source-level function matches.

Same shape for `seq_new` (`lean_single.sio:10501`, dispatched `13832`), `observe`, `measure`, `uncertainty_of`, `variance_of`, `require_confidence`. Every codegen intrinsic has a dispatch site but no source definition.

**Critical finding — the typechecker has no pre-registration mechanism for these.** The lookup at `check.sio:19734-19746`:

```sio
ExprKind::ExprIdent => {
    if !c.env.has_binding(e.name) {
        let fn_sig_id = fn_sig_table_find(c.fn_sigs, e.name)
        if fn_sig_id >= 0 {
            return (c, ty_fn(fn_sig_id))
        }
        let c_err = c.report_undeclared(e.span, e.name)
        return (c_err, ty_error())
    }
```

checks `env` then `fn_sigs`. There is **zero** `fn_sig_table_insert` call anywhere in the codebase:

```
$ grep -rn "fn_sig_table_insert" self-hosted/
(no matches)
```

So the typechecker has no record of any codegen intrinsic. Every intrinsic call (`measure`, `acknowledge`, `observe`, `seq_new`, `uncertainty_of`, `variance_of`, `require_confidence`) fails at typecheck with E137 — even though the codegen layer below would handle it correctly.

The defect is **not in the codegen**; codegen for these intrinsics works. The defect is in the **typechecker**: missing `fn_sigs` pre-registration for intrinsics. The fix lives on the self-hosted/check or self-hosted/compiler lane that owns `check.sio`.

The fix: add a `setup_intrinsics(c: &!) -> Checker` pass at checker construction, populating `fn_sigs` with type signatures for every codegen intrinsic. Each entry needs name + arity + param types + return type. Roughly:

```sio
fn setup_intrinsics(c: &!Checker) ->Checker with Mut, IO, Panic, Div, Alloc {
    var c2 = c
    c2 = c2.fn_sig_table_insert("measure", ty_fn_signature_2(...))
    c2 = c2.fn_sig_table_insert("acknowledge", ty_fn_signature_2(...))
    c2 = c2.fn_sig_table_insert("observe", ty_fn_signature_3(...))
    c2 = c2.fn_sig_table_insert("seq_new", ty_fn_signature_0(...))
    c2 = c2.fn_sig_table_insert("uncertainty_of", ty_fn_signature_1(...))
    c2 = c2.fn_sig_table_insert("variance_of", ty_fn_signature_1(...))
    c2 = c2.fn_sig_table_insert("require_confidence", ty_fn_signature_1(...))
    c2
}
```

The exact type signatures need to match what the codegen layer produces (return type, effect set, knowledge type parameter). One existing reference point: the codegen-intrinsic dispatch sites know the expected arity (`compile_acknowledge_call_x86` at lean_single.sio:10249 takes `call_tok: i64` and emits a `tc_arity_error` for anything other than 2 args). That arity check is the contract the typechecker stub must match.

## Side effect — same fix unlocks three more tests

The `acknowledge`/`observe`/`seq_new`/`measure` fix is **not just for `rapamycin_kaxi_fuse_prior.sio`**. The same typecheck stub passes unblock two more tests currently failing with the same E137:

```
$ ./bin/souc check tests/run-pass/observe_contraction.sio
error[E137] in run-pass/observe_contraction::main at 659..666: use of undeclared variable
   = name observe
error[E137] in run-pass/observe_contraction::main at 715..726: use of undeclared variable
   = name acknowledge
error[E137] in run-pass/observe_contraction::main at 848..855: use of undeclared variable
   = name observe
error[E137] in run-pass/observe_contraction::main at 904..915: use of undeclared variable
   = name acknowledge

$ ./bin/souc check tests/run-pass/knowledge_acknowledge.sio
error[E137] in run-pass/knowledge_acknowledge::main at 205..216: use of undeclared variable
   = name acknowledge
```

`observe_contraction.sio` is the central M1 math-review witness for the Bayesian posterior contraction (`observe(k, y, uncertainty: s)` does normal-normal conjugate update). `knowledge_acknowledge.sio` is the test that `acknowledge()` extracts value from `Knowledge<T>` without requiring Epistemic effect. Both are tagged `@ run-pass` and currently fail in the same way — same E137 from the same dispatch site, on codegen intrinsics without typecheck stubs.

**This changes the value of the fix.** Without the side effect, the typecheck-stub fix unblocks one dissertation test (`rapamycin_kaxi_fuse_prior`). With it, it unblocks three: the dissertation test plus the two epistemic-stdlib witnesses. The fix should be sized for three tests, not one.

## Why these look like one cause but are not

At the typecheck dispatch site (`check.sio:19734`), both causes produce the same diagnostic: E137 with help "declare the variable before use, or import it from another module." The typechecker does not distinguish:

- "name exists in source but is private / not re-exported" (print_i64)
- "name exists at codegen but has no typecheck stub" (acknowledge, seq_new, observe, measure, uncertainty_of, variance_of, require_confidence)

But the fixes are on different files in different lanes:

| Cause | Where | What | Lane |
|---|---|---|---|
| A. print_i64 | `stdlib/plot/bar.sio`, `stdlib/metrology/calibration.sio` | add `pub fn`; re-export from common module | grok-cli2 (stdlib/darwin_pbpk) |
| B. acknowledge/seq_new/observe/measure | `self-hosted/check/check.sio` | pre-register intrinsics in `fn_sigs` during checker setup | self-hosted/check or self-hosted/compiler |

You could solve Cause A with a Cause-B-style fix (register `print_i64` as a typecheck stub) — but that would compile while hiding the visibility bug. You cannot solve Cause B with the Cause A fix because these names have no source definition. The two causes are not collapsible into one without either hiding a bug (Cause A → Cause-B-style fix) or pretending a missing fix is the right one (Cause B → Cause-A-style fix won't work).

## Defect class — same as today's four PRs

The user's framing: "a plausible value where an absence should have been declared." E137 here is exactly that shape, on the diagnostic side:

- **Cause A**: `print_i64` is plausibly a function (we see it in source). Its **absence** of `pub` is undeclared.
- **Cause B**: `acknowledge` is plausibly a function (we see codegen handling it). Its **absence** of typecheck registration is undeclared.

Same shape as:
- pin_count returning 0 where −1 (declared absence) was needed (#1830)
- CALL refusal returning unsigned where a name was needed (#1825)
- fabrication returning boolean where a visible component was needed (#1829)
- reclamation returning unwitnessed where a witness was needed (#1823)

The compiler's name-lookup at `check.sio:19734` is the analog of `pin_count`: it returns "undeclared" when the honest answer for some cases is "registered-as-intrinsic" (for codegen intrinsics) or "exists-but-private" (for stdlib functions). The typechecker has no way to say either.

## The pattern (one error code hiding multiple causes)

Today is the third time:

1. **lower_array**: looked like one cause, was two. Different bugs surfaced as the same symptom. Resolved by separation: each cause got its own lane owner.
2. **rc=182**: looked like one cause, was one. Four independent measurements converged. The "is this one or two" question was settled by measurement, not by inspection.
3. **E137** (this doc): looked like one cause, was two. Distinct fixes on distinct lanes.

The pattern: when an error code fires in a test and the test does not point at a single line of source, the question "is this one cause or several?" is worth asking **before** assigning the fix to a lane. The cheapest way to ask it is to compile the failing test against the current compiler, capture the byte offsets and the failed names, and check whether the names are even the same kind of thing. If `print_i64` is "function exists with wrong visibility" and `acknowledge` is "function exists only at codegen," they are not the same cause even though both say E137.

The cost of getting it wrong is the rc=182 mistake applied to lane assignment: a fix landed in the wrong file, sat there unmerged, and was eventually re-routed. The cost of asking is one `bin/souc check` run per failing test, which the lane already does.

## Files and references

- Compiled E137 sources:
  - `examples/dissertation_scenario_gate_demo.sio` — `bbb_voi_print` at byte 4885..4894, name `print_i64`
  - `tests/run-pass/rapamycin_kaxi_fuse_prior.sio` — `main` at 1856..1863 (`seq_new`), `main` at 2340..2351 (`acknowledge`), `kaxi::kaxi_fuse` at 1013..1024 (`acknowledge`)
  - `tests/run-pass/observe_contraction.sio` — `main` at 659..666 (`observe`), 715..726 (`acknowledge`), 848..855 (`observe`), 904..915 (`acknowledge`)
  - `tests/run-pass/knowledge_acknowledge.sio` — `main` at 205..216 (`acknowledge`)

- Compiler source anchors (read, not modified — lane discipline):
  - `self-hosted/check/check.sio:19734-19746` — E137 dispatch (env → fn_sigs → report_undeclared)
  - `self-hosted/check/check.sio:12770` — E137 message string
  - `self-hosted/check/check.sio:19745` — the only call site of `report_undeclared`
  - `self-hosted/compiler/lean_single.sio:10249` — `compile_acknowledge_call_x86`
  - `self-hosted/compiler/lean_single.sio:10501` — `emit_seq_new_x86`
  - `self-hosted/compiler/lean_single.sio:13022` and `:33505` — intrinsic dispatch sites
  - **Zero `fn_sig_table_insert` anywhere in `self-hosted/`** — confirms no pre-registration exists

- Stdlib source anchors (grok-cli2's lane, read-only here):
  - `stdlib/plot/bar.sio:301` — `fn print_i64` (not pub)
  - `stdlib/metrology/calibration.sio:322` — `fn print_i64` (not pub)
  - `stdlib/metrology/calibration.sio:332` — `fn print_i64_abs` (not pub, related)

## Status

- Diagnosis: two causes ✓ (this file)
- Affected tests: 4 (dissertation_scenario_gate, rapamycin_kaxi_fuse_prior, observe_contraction, knowledge_acknowledge) ✓
- Cause A fix spec: add `pub` + re-export; lane = grok-cli2 ✓
- Cause B fix spec: pre-register intrinsics in `fn_sigs`; lane = self-hosted/check ✓
- Cause B side effect: unlocks 3 tests, not 1 ✓
- Pattern note (one code / multiple causes): recorded ✓
- This lane's contribution is the diagnosis. The fixes live on other lanes and were not applied.
