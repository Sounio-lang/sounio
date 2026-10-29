# Schema2 diagnostics for empty externs and global DATA

This is a diagnostic-only change. It does not alter Worker proof, LCE proof, lowering, native emission, admission, or any default feature activation. Schema version, raw fields, counters, BEGIN/END framing and instruction/argument/name/bitmap limits remain unchanged.

The producer's `IrFunction` table contains both functions and strategy99 global DATA. For globals, `param_count` stores the BSS byte offset; `bss_size` stores the extent. `returns_float` is reused as an initialization flag, `prof_counter_id` carries initializer data, and `hyper_algebra_kind` carries element information. Treating their offset as a function arity/vreg count prevented a complete diagnostic capture. These fields are now serialized faithfully without interpreting them as callable authority.

## Exact live evidence

Raw04 remains `a4f5e8d655754202493bbb04d8fbb15890b10e8fc861ac3dc60438452704c333`. Slurm19717 completed0 in27s with a private C++23 ptrace observer: all19 WIDE,35 direct and197 QPU analyzer SRet returns were captured; C++ independently matched each public aggregate from the same successful compiler execution. Source/compiler guards passed.

The two descriptors that caused the previous formatter reason4 were:

- Direct fi2, `dlopen`: no body, registers or labels; param_count2; all64 param_regs=-1; strategy0 and ordinary empty signature metadata.
- QPU fi176, `QUANTUM_MAX_AMPLITUDES`: no body, registers or labels; strategy99; BSS offset8 and extent8; all64 param_regs=-1. Reused fields were returns_float1, prof_counter_id1024 and hyper_algebra_kind8. This is DATA, not a function with eight active parameters.

Descriptors were observed at the real preseal LCE entry. Both entries have no body and are rejected by LCE before mutation, so their signature/data metadata is unchanged at the later source formatter. This is descriptor evidence; it is not a substitute for the imported full arena that the formatter refused.

The same observation confirms WIDE's literal and previously published hoisted cores both return LCE status10. The module aggregate is exactly19functions/18refused/2entries/0eligible. The independent must-definition investigation is separate; this patch does not weaken that refusal or assert an origin for the missing definition.

The observer uses the real SRet prologue and validates module counts, function names, return destinations, all per-function audit returns and the public aggregate. Early scanner/ABI attempts were refused and retained. A separate FULL-code fingerprint was rejected by its runtime ABI and was never trusted as a formatter descriptor source. The accepted descriptor capture comes from the validated analyzer hook.

## Narrow diagnostic profiles

The production formatter and its copied component helper share the same body:

- Defined functions retain the original active parameter-register bounds.
- Empty externs require strategy0, zero body/register/label counts, arity0..64 and every fixed param_regs slot equal to -1.
- Global DATA requires strategy99, zero body/register/label counts and every fixed param_regs slot equal to -1. Offset and extent must be nonnegative, and offset must not exceed INT64_MAX minus extent. The offset is never iterated as an arity.
- Other reused DATA fields are not normalized or restricted speculatively. Their exact values remain in the existing schema.

The C++23 parser classifies DATA separately, enforces the same shape/span conditions and keeps the original framing, census and completeness checks. Empty DATA and extern records cannot contain invented parameter vregs. Global records with initializer instruction bodies remain refused by this deliberately bounded diagnostic profile.

The fixture now accepts `TIME_BIN` for the preserved GNU time binary; the default remains `/usr/bin/time`. Flags, assertions and deadlines are unchanged. The historical `nv2_trace_full.patch` remains byte-identical, SHA256 `5281be3be5d2556b077f60c48f2f2fbf3102a8feb2eefb8630d81141b6b22ab6`.

## Component qualification

Slurm19735 completed0 in15s using raw04 for checking, exact seed9cf for native component producers, and private-v3 GCC13/C++23. The immutable raw04 source was cloned and only the five changed diagnostic/test files overlaid. No bootstrap or whole modern frontend was repeated.

Modern check passed. Both historical-helper and candidate native producers compiled without explicit compiler errors and executed successfully. Historical helper bytes were substituted into the otherwise identical new probe for the paired baseline:

- Before: extern1/2 and global DATA were refused as reason4; extern0 passed. Empty stubs with invented register/label metadata were accepted by the old formatter/parser.
- After: extern arities0/1/2 and DATA at offsets8,4096 and the maximum valid extent passed in both source and prepared stages. A raw DATA profile with negative initialization flag/initializer and distinct element metadata was preserved exactly by the C++ oracle.
- Defined missing parameter registers, invented empty-stub registers/padding/labels, negative DATA offset/size, overflowing span, DATA body and invented DATA vregs/labels were refused.
- Independent parser mutations cover negative/overflowing DATA span, DATA body, strategy laundering, invented inactive/active vregs, empty frame/label metadata, and partial DATA captures.
- Every existing source/prepared payload, legacy output, malformed args/names/regions/budget, missing/wrong stage, unknown record, truncation, duplicate group and census mutation control remains passing.

`component/GATE_PHASES.tsv` and `component/BEFORE_PHASES.tsv` retain every expected/actual result. Full stdout/stderr, source, producers, parsers, all guards and earlier failed observer attempts are in the durable archive. This qualifies the component and the exact shared production helper body. A rebuilt compiler's real CLI capture is still required to close the imported formatter boundary; raw04 itself was not changed.

## Provenance and custody

Isolated worktree: `/workspace/sounio-stdlib-wave4-trace-empty-20261029`, base `ea1b9b9b5e1cb837285498e192167712cdbb78cf`, own devpass branch/claim. The older compiler worktree and all capacity WIP were preserved.

Production codegen candidate SHA256: `429baad26ee28d4dbc798e849b0f6eeccc672f388a1b56ca1dfcae942a6e342a`. Before: `0e370fddc5c85147bc5e8215f4cc1e6b8ac4adecf8f172d3fc28db5a36ac1ac7`.

Slurm19736 completed0 in10s. Archive creation required the real Ceph mount, verified the base raw04 archive, audited the tar listing, read the archive back and rechecked candidate source/inputs:

```
/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/trace-empty-export01/trace-empty-component-and-live-metadata-v1.tar.gz
SHA256 223af965b6d04d914a150e22dcb6309567e22a086f48270b56d0e24f05e2f479
```

The archive binds its base raw04 sources/seed/SDK through the already verified archive SHA `22e7f121aff116216da7f1acc35221acdcbde5d92c63b36aa60b9d89f0e43b60`. Committed compact manifests bind the complete guard/trace files; no trace was truncated into a claimed full arena.
