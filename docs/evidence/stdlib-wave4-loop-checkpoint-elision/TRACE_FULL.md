# Complete NV2 arena trace diagnostic: stage-bound schema 2 artifact

The diagnostic patch is prepared for root review and **has not been applied**
to a production codegen file or compiler ELF:

`tests/fixtures/wave4_loop_checkpoint_elision/nv2_trace_full.patch`

Patch SHA-256:
`5281be3be5d2556b077f60c48f2f2fbf3102a8feb2eefb8630d81141b6b22ab6`.

Its base is root `fa485cc07e5596573eb99807184ccb47878fa198`,
`self-hosted/native/codegen_x86_linux.sio` SHA-256
`d3380a0826be9547204e8cc53d35c45e6fd1e010f421172d64c80a1e6507e01d`.
Candidate SHA-256:
`0e370fddc5c85147bc5e8215f4cc1e6b8ac4adecf8f172d3fc28db5a36ac1ac7`.
`git apply --check` against the root codegen file succeeds, without mutation.

The schema-1 artifact remains in historical commit
`8aee6eaa3ff8c08ba44423ca1f4cefd8de81a334`. The current patch supersedes it.

The new opt-in is exactly `SOUNIO_NV2_IR_TRACE_FULL=1`. The existing
`SOUNIO_NV2_IR_TRACE` and legacy trace body are retained. Removing only the
added helper, named imports and two stage-hook calls reproduces the **entire original
codegen source byte for byte**. See
[trace-full-stage2/LEGACY_SOURCE_CONTROL](trace-full/LEGACY_SOURCE_CONTROL).
A standalone false-branch control also preserves its legacy output exactly.
This is structural/component evidence, not a rebuilt production-ELF replay.

## Schema 2 and completeness

The writer validates the entire capture before BEGIN. Refusal emits only
`NV2_IR_FULL_REFUSED 1 reason function instruction`, with no BEGIN/END pair.
A diagnostic refusal does not alter the enclosing compiler's exit status;
a consumer must require the complete versioned pair and matching counts.

BEGIN carries version 2, stage code, function count, total instruction/raw-argument/name-byte
counts, float bitmap word count, argument-pool length, arena capacity and
name-pool count. END repeats the version/stage and actual emitted
function/instruction/argument/name counts. The C++23 parser rejects truncated
captures, duplicate groups, count mismatches and trailing full-trace rows.

Records contain:

- Every function's region slot/generation/status/seal/length/capacity,
  instruction/register/label/parameter counts, strategy, return
  float/Knowledge/fixed-array metadata, profiling/algebra/sret/BSS/ref/f128
  descriptors, full parameter array and complete float bitmap.
- Function and return names as length-delimited unsigned bytes.
- Every instruction's function/index/physical slot, exact opcode enum tag,
  destination/sources, both immediates, label/callee/field, flags, exact
  binary/unary enum tags, interned name id, raw argument count and pool base.
- Every raw argument read with `ir_arena_arg_at(slot,index)`, never from the
  unpopulated `IrInstr.call_args` of an arena load.
- Every instruction name as bytes. There is no ambiguous `op=other` or lossy
  string escaping.

Integer immediates and f64 payloads use two unsigned 32-bit limbs, preserving
signed-minimum integers, negative zero and NaN payloads exactly. The f64
bitcast first binds an i64 local.

Explicit bounds: 131072 instructions, 524288 raw arguments, 64 parameters or
arguments per site, 384 bytes per name, at most 65536 float register bits.
Function and individual-instruction bounds also honor `IR_MAX_FUNCS` and
`IR_MAX_INSTRS`. All region/pool extents are checked before access. Unknown
opcode tags are recorded exactly; the formatter does not certify semantics.

## Current stage-bound component qualification

Slurm **19623 completed 0** in eight seconds: modern check 0, seed producer
0, and the published schema-2 stage gate 0. Source selection emits the fixture's
six-function module; prepared selection emits its distinct seven-function
module after a simulated clone. Every byte/field/census and the same exact
signed-minimum, negative-zero, NaN and slot-argument payloads pass C++23.

The stage selection is explicit:

- `SOUNIO_NV2_IR_TRACE_FULL_STAGE=source` captures in preview before
  Worker preparation can fail. The prepared hook suppresses full output.
- `SOUNIO_NV2_IR_TRACE_FULL_STAGE=prepared` captures only at the existing
  prepared trace site. Unset/empty stage selects this same default.
- Any other stage emits refusal reason 10 at source preview and no full group.
- Source is encoded as 0 and prepared as 1 in both BEGIN and END. Neither is
  a certificate. C++23 requires `--check source FILE` or
  `--check prepared FILE` (similarly for `--probe`).

Both stages reject ten mutations and five malformed/bounded inputs. Wrong
stage selection, a concatenation of both stages, schema-1 input and an omitted
stage selector are also rejected. Default-prepared output is byte-identical
to explicit prepared selection; FULL absent retains the control output with
either stage environment value. These are **stage-routing/formatter component
results**, not an actual Worker IR capture or a Worker proof result.

The new records are in [trace-full-stage2](trace-full-stage2). The previously
published schema-1 records are retained below as historical evidence.

The initial uncommitted schema-2 packaging attempt omitted helper/hooks due
to a Perl generation-variable scope error. Its inverse guard refused it.
It was not applied or committed. The accepted candidate was regenerated,
compared byte-for-byte against a separately constructed 761370-byte candidate,
inverse-stripped to the original source, and checked with git apply. The
obsolete packaging hashes 6a60.../f405... are not accepted artifacts.

## Historical schema-1 component qualification

Slurm **19614 completed 0**:

- Modern raw 8abe checks the isolated formatter probe, verdict 0.
- Seed 9cf compiles it, verdict 0; its native ELF emits the complete capture.
- Independent C++23 checks every field/census, signed-minimum and exact
  negative-zero/NaN payloads, float bitmap bits, and exit arguments from SLOT.
- Truncation, duplicate instruction, wrong total, missing argument, altered
  payload, malformed name length, missing float word, unknown record and
  duplicate end are rejected.
- Corrupt argument-pool/name/region/parameter metadata and total instruction
  budget produce explicit refusals. The parser rejects each as a full capture.
- Original attempt 19612 checked 0 but seed compile refused the unavailable
  `bits_to_f64` builtin. It remains preserved. The accepted fixture uses a
  fixture-only typed pointer reinterpretation. Product trace code uses only
  `f64_to_bits`.

Slurm **19617 completed 0** reruns the published `nv2_trace_full_gate.sh`
against these exact component/oracle ELFs. Logs are in
[trace-full/published-gate](trace-full/published-gate); producer/source/ELF
manifests are in [trace-full](trace-full).

Raw:
`8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932`.
Seed:
`9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a`.

This is **formatter-only component qualification**. No complete imported
quantum-benchmark arena has been captured yet.

## Preserved capture-route failure and remaining gate

The old NV2 trace omits control opcode distinctions, labels, immediates,
operators and full arguments. It cannot be reconstructed into trustworthy IR.

Public frontend/serializer component 19595 compiled with seed 9cf, but both
inputs failed before returning IrModule: the projection had unresolved
parameters/locals in lowering, and the imported target had linear-use errors
in qasm_tape. Those failures are preserved in
`/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/loop-arena01`.
They are not successful captures. SOIR v6's CLI also emits only a summary
and may refuse signatures or images beyond 128 KiB.

After root review and an authorized compiler build, this diagnostic can
capture the full source module before Worker preparation, or the complete
prepared module at the existing native trace callsite.
It changes no IR and performs no elision. Real helper metadata, including
parameter counts, remains an explicit prerequisite to qualifying the
optimization on the benchmark.

The published schema-2 gate is reproduced with:

```sh
bash tests/fixtures/wave4_loop_checkpoint_elision/nv2_trace_full_gate.sh \\
  COMPONENT_ELF CPP23_ORACLE \\
  docs/evidence/stdlib-wave4-loop-checkpoint-elision/trace-full/results/valid.stdout \\
  FRESH_OUTPUT
```

The schema-1 log argument is intentionally an input that the current parser
must reject. The gate requires a Slurm allocation.
