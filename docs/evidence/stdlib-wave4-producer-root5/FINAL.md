# Qualified private x86 producer: return boundary, bounded Option tuple joins and structural publication

The final source is self-hosted/compiler/lean_single.sio SHA256
815661329ab8a1e65c851982b1a4b4b4ae504d63c6dcea911c95602f5a8863b0.

Generation1 private ELF:
d8e8eb31a9eec0075694dbed0a4a1c667d6b7ac3baaf8887f8734b646770372a.
Generation2 private ELF:
f2ab364356061eb9f092741e0c0e2ecea0c02018745a5b83e8b9926e547a57f8.

The qualified generation2 remains at
/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-final-transport04/results/generation2-producer.elf.
No repository/shared seed was replaced. This is a legacy x86 producer qualification, not a completed modern compiler bootstrap.

## Resulting behavior

A comma ends a bare return operand in an x86 match arm. Unit returns and nested Unit matches preserve actual effects; nonUnit bare returns still refuse. This matches the modern parser's existing at_expr_start rule for comma. No additional delimiter or grammar change is made.

A match can reconcile a partial literal tuple containing actual builtin None with an existing concrete sibling literal tuple containing actual builtin Some. This is bounded representation inference, not a change to ty_eq. Every inferred Option payload is an explicitly allowed one-word primitive (integer/f64/bool with canonical hash0) or a concrete Box of an allowed scalar/registered nominal. References, raw pointers, function values, Knowledge, direct wide records, unknown types and boxed unknowns are refused. Other tuple fields are within the bounded primitive profile. A cell containing a nested tuple cannot participate in a new coercion.

The emitter records facts only AFTER actual None tag/payload zero stores or Some tag/payload emission. A token name or ExprTy11/hash0 alone is insufficient: Ok/Err share that old unknown representation. Qualified, local/global/function-shadowed or enum-ambiguous constructor forms cannot acquire a fact. Ordinary typed Option calls can continue passing when their concrete types already match; this is preexisting acceptance, not new inference.

An immutable proof record binds the exact literal tuple hash/arity and actual emitted element row, so a packed-key collision cannot authorize a fact. Consumers require the exact whole-expression span/type/hash. Variable loads, calls, postfixes or stale child facts cannot mint a new None fact. Existing type/cache equality functions are unchanged.

Facts may be transported through existing if/if-let/match joins only if all reachable value arms have strictly identical kind/hash/layout and equal emission-proof shape, or if exactly one value arm reaches the join. This is a meet of existing facts. It performs no None coercion in if, guesses no reachability and never transports through a nested tuple cell.

The proof arena has4096 physical records and1024 optional records per frame. Top-level function frames do not accumulate records. Closures append within a saved frame and roll back without overwriting caller/first-arm IDs. Exhaustion drops optional facts; a coercion requiring an unavailable fact then refuses. An unused concrete tuple does not fail merely because its optional fact was dropped.

New strict x86 structural flow/join/exhaustiveness/nonUnit errors now block ELF publication even in imported functions. Actual primary parser fallback also forces publication failure. The old tc_mark_failed imported-policy function and tc_error warning function remain byte-identical. The broad legacy warning policy is not changed.

## Final terminal Slurm receipts

All jobs below are COMPLETED Exit0 on dl380,1CPU/8GiB. Individual compiler/native/oracle outcomes are retained; collector exit alone is not used as correctness evidence.

|Job|Gate|Observed outcome|
|---|---|---|
|19838|Original104 fixtures, generation1 and generation2, selfcompile and AArch64 source guard|104/104 twice; selfcompile0;148 AArch64 function bodies unchanged; original baseline45 failures preserved; oracle corruption controls refused|
|19839|Real producer profiles, flat transport, raw observation cells, closure frames and budgets|Flat13rows/f64bits/recursive Box/wide Box exact;4 tag/payload observations exact;18 branch-transport rows exact after clobber512;5000 functions compile/native;1022 fact prefix accepts,1023 coercion refuses,1100 unused facts accepts; fresh recovery compilation/native succeeds|
|19840|Unchanged presence and metadata boundary components|211 rows, all69 channels/fields,0 mismatches;34 metadata cases,0 failures/0 OOB|
|19843|Original17 main/imported reductions and executed grammar fixture|Expected compiler statuses and no invalid ELF; Unit/comma/nested/valued native effects exact|
|19844|Three exact SOURCE237 consumer bodies with explicitly labeled typed seams|Before ccfa3 arm errors; final0 errors/native0/C++27 rows exact after clobber512|
|19845|Actual emitters and extended proof component|None24 bytes and Some41 bytes exact; actual tag/payload initialization verified;48 proof/span/family/frame/cap/meet controls pass|
|19848|Shared source guards and independent negative oracle controls|ty_eq/tc_mark_failed/tc_error unchanged;7 value/NaN/missing/duplicate/payload/opcode/transport mutants refused; real closure preserves first-arm proof|
|19853|Final source/ELF pin and external custody|All admission assertions and pins pass; archive readback SHA verified|

The profiles distinguish bounded refusal from grammar limitations. native_cells.sio uses pointer arithmetic unsupported by this producer and is retained as a refusal. The separate native_cells_array.sio uses a fixed two-cell observation view and passes all original tag/payload expectations. stale_call_none is a legitimate preexisting concrete typed call; stale_call_unknown and stale_variable_none cannot reuse constructor facts.

## Exact-consumer component limits

The checked-in ledger in final/exact/EXTRACTIONS.json binds every extracted body to its full SOURCE237 source-file hash and line. Bodies are byte-exact, including the recursive checker_collect_closure_param_list_inplace helper. Actual parser Expr/ExprList, TypeEntry and FnParamList declarations are used.

The Checker pointer type-lowering dependency and the Lowerer receiver are explicitly labeled typed component seams. The Checker seam returns controlled actual TypeEntry records; the Lowerer seam uses the exact sanitizer method. This does not certify the actual8MiB Checker or complete Lowerer storage ABI. A subsequent single fresh modern bootstrap, actual CLI/closure/loader and the historical12 original source gates remain mandatory.

19830 preserves the earlier exact-consumer failure: the first bounded candidate cleared two joins and conservatively lost the Some fact through the parser's inner compound joins. The final strict proof meet closes that third exact body. Its driver-only reference-to-raw-pointer cast is part of the explicitly typed seam, not a consumer source change.

## Preserved open failure and predecessors

The original combined nested tuple native fixture and its expected output are unchanged.19818 returned0/604 and0/593 instead of42/1 and-9/0 after stack clobber; its C++ oracle failed. Final inference explicitly refuses traversal of that pointer cell. No heapification, ABI repair or expected-value weakening is included. See19818/OPEN_NESTED_TRANSPORT.md and immutable a554e67b.

19801/19804 establish grammar and contextual-join regressions and imported error suppression;0e84241d preserves them.19803 duplicate setup retry failed before compilation.19812 separately qualifies grammar/fatal latch before inference.19820/23/26/27/32 qualify the predecessor flat source; they are not substituted for the final source gates.19824 and19842 are preserved technical setup failures (unsupported observation fixture expectation; overescaped private-main rename), not successful semantic tests.

The failed ROOT5 main job19789 has no RAW_READY and its ELF is never admitted here. No new modern bootstrap, historical scientific replay or61B QPU run was performed.

## Full immutable custody

Archive:
/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/root5-producer-flat-final/producer-matrices-sdk-receipts.tar.gz

SHA256629ba4fc442157de7403d28a03133520a1bae077e8f57fd89a030e30bd741e9d.

It retains complete final snapshots, actual SDK copies, generated stress inputs, source/seed/ELF manifests, all compiler/native logs and ELFs, C++ references, exact body ledgers, Slurm recipes and full protocol controls. Repository evidence is a compact review set; this archive is the complete custody copy.

