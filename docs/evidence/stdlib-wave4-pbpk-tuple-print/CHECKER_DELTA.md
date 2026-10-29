# Canonical i64 binding stamp: component-qualified, CLI integration pending

Root approved a narrower implementation than the initial proposal: only exact canonical `TyI64`, with no lowerer changes and no f64 reification. The positive f64 and nested controls in the baseline establish preserved behavior; they do not expand this implementation's whitelist.

`check::binding_stamp::binding_stamp_checked_i64` writes only the original `Box<StmtList>.head.ty`. Both statement-list spines snapshot environment count/scope and error count, check the statement once, then call the helper before loan finalization or scope pop. Their statement return type and `last_literal_kind` handling are unchanged. No RHS is checked or evaluated by the helper.

Admission requires all of the following:

- No existing annotation, a let or var, and a nonempty bounded name.
- No prior or current errors, no sticky checker error, no environment overflow.
- Exactly one new binding at the pre-check count, with the same current scope, name and mutability as this original statement. Lookup by span or by a previously visible name is not used.
- A complete canonical `ty_i64()` shape: exact kind, zero name including its full buffer, no inner or tuple elements, canonical array/function sentinels and all algebra/unit/refinement/epistemic/robustness/ontology/noise/provenance/validity fields. The epsilon and validity floats are compared by bits, including signed zero.

The AST annotation has the original statement span, the builtin one-segment i64 path and no wrappers, generic parameters, effects, Knowledge or refinement data. Publication uses a normal managed field assignment. All other statement fields and links remain untouched. Existing authored annotations are never overwritten.

## Qualification

- 19763: helper and component modern checks0. Modern compile SIG139 at `lower_array: seed_begin`, before an ELF. This remains an unresolved producer failure and is not a stamp refusal or a passing gate.
- 19765: legacy invocation from the wrong working directory refused unreadable imports; preserved.
- 19766: corrected import directory exposed the test-only `bits_to_f64` name unsupported by the legacy seed; preserved. The control was explicitly changed to the equivalent source literal `-0.0`; production helper unchanged.
- 19767: legacy compile0, no explicit compiler errors, then native launch126 because the seed output lacked executable permission; preserved.
- 19768: the same ELF after chmod, modern check and native0. No recompile or binary rewrite.
- 19771: extended component modern check0, legacy compile/native0, 7 seconds.
- 19772: the published, regenerating gate passed in9 seconds with modern helper/component checks, legacy compile/native and source/tool/ELF guards0.

The extended component covers 36 noncanonical TypeEntry variants, including every metadata field, name padding, signed-zero validity, Error/Unknown/f64/ref/Unit/wide types and both Worker capabilities; 12 binding/error/state guards; same-name shadowing; mutable binding; existing annotation; alias visibility and captured Block visibility; repeated invocation; and 5000 intervening allocation calls after publication.

Worker controls use the actual source-profile prefix extracted from `native/worker_proof.sio` up to the unique `WorkerLowerBinding` boundary. Function bodies are preserved. A scalar RHS passes before and after annotation; array, struct, unresolved call and unresolved identifier remain refused even when adversarial post-check metadata supplies i64. This is a source-profile component, not Worker issuance or final-IR/native integration.

The component's second invocation demonstrates helper idempotence. It does not replace a complete second run of the real checker. Similarly, survival through helper return and allocation churn is a concrete native publication control, not an audit of every frontend arena-reset path.

## Still required before claiming the PBPK fix

Build a compiler containing this delta with the repaired producer, then run the unchanged original PBPK with its1200-second compile deadline and preserve any later dependency11/SIG139 separately. Run the tuple matrix and exact signed/f64 C++23 oracle on that compiler, both loop modes; run complete same-AST and fresh-AST checker controls in both spines, including loans/effects/errors and frontend lifetime paths. Existing Worker source/final-IR/forgery gates must retain their results.

No new compiler bootstrap, lower edit, original PBPK edit, print workaround, public type authority or relaxed failure rule is included here.
