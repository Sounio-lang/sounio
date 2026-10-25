<!-- docs:meta
topic_id: repo.docs.spec.2026-10.06-execution-memory-and-targets
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.2026-10.06-execution-memory-and-targets
-->

# 06. Execution, Memory, Calling Conventions, and Targets

Evidence catalog: [06-evidence.tsv](06-evidence.tsv).

Source baseline: `f5b2c5850be9f3d3995719ffcade44a6d851658a`, inspected on
2026-10-16 in the shared specification worktree. This chapter records source
research, not a compiler execution receipt. No compiler build, runtime gate,
numerical oracle, or Lean build was executed for this chapter. Existing reports
are identified as reports; their historical results are not fresh measurements.

## 6.1 Scope and interpretation

This chapter separates the meaning of executing a Sounio program from the
representation used by a particular compiler and target. It covers entrypoint
selection, lowering, sequential evaluation, storage, internal calls, foreign
calls, failure reporting, and target admission. The type and ownership chapter
owns the static rules for moves, linear resources, and borrows; the effects
chapter owns effect checking; the numeric chapter owns arithmetic formats and
algebraic laws. None of those contracts can be inferred merely from a machine
instruction emitted here.

Each numbered rule has exactly one standing, also recorded in the companion
catalog:

| Standing | Interpretation |
|---|---|
| `source-observed` | A descriptive contract of the named source path. It is not a claim that a shipped binary has that behavior. |
| `specified-contract` | An existing project requirement, distinguished from evidence of implementation. |
| `library-contract` | A contract of a named library or tool, without general language promotion. |
| `target-contract` | A descriptive contract restricted to a target and route. |
| `open-decision` | A missing language decision or acceptance boundary. Candidate alternatives are not approved semantics. |

In this chapter, **reachable** means a named entrypoint has a source call path
to the implementation. **Emittable** means that implementation contains a writer
for the requested representation. **Executable** requires a fresh receipt from
running the resulting artifact on the stated target. **Conformant** additionally
requires comparison against an adopted rule and its discriminating witnesses.
The existence of a source file establishes none of the latter three by itself.

An execution observation comprises normal return or abnormal termination,
observable I/O, externally visible mutation, foreign interactions, and relevant
scientific channels. An ABI describes where values travel; it does not decide
their mathematical meaning, ownership, or scientific validity.

## 6.2 Compiler identity, resolution, and bootstrap

### Public entrypoints

`EXEC-001` — `source-observed`: the ordinary public verb route is `bin/souc`
to `bin/madaros`, then the selected raw modular compiler. The launchers prefer
explicit raw overrides, the host-specific artifact when available, the local
`artifacts/self-hosted/madaros`, and the committed Linux prebuilt. The Darwin
host candidate is `artifacts/self-hosted/madaros-arm64-macos`, admitted by
`sounio_madaros_host_raw` only on Darwin arm64 with an executable Mach-O arm64
artifact. This is resolution priority, not source freshness verification.

The shared resolver interfaces are not interchangeable. `resolve_souc.sh`
resolves a public compiler command for callers, while
`_sounio_resolve_madaros_bin` also accepts `MADAROS_BIN` and can return a wrapper
or a raw binary. The launchers have their own raw-binary selection. A gate must
record the final executable, not merely the variable or wrapper it supplied.
The launcher's `_is_elf` predicate checks executable status and excludes a
shebang; its name does not mean that it verifies an ELF architecture. This is
why a usable native Mach-O override can pass that predicate.

`EXEC-002` — `source-observed`: a nonempty explicit raw override that is
missing, a directory, nonexecutable, or a script is refused by the public
launcher rather than silently replaced by a committed compiler. The launcher
reports `E-SRB-001` and status 78 for this failure. The resolver library has a
related executable-override check with its own message; it is not a promise of
one uniform diagnostic across all resolver entrypoints. See
`compiler_override_fail_closed_gate.sh` for positive and adversarial controls.

`EXEC-003` — `source-observed`: raw positional invocation
`bin/souc source.sio output` selects the preserved `lean_single` route; it does
not select modular Madaros merely because Madaros is the default for verbs.
The bare form refuses a flag-looking second argument, avoiding an output file
literally named `-o`. `SOUNIO_SOUC_ENGINE=lean_single` selects the seed explicitly;
the wrapper translates supported verbs to its positional interface. If no
Madaros candidate is available, the normal wrapper can visibly fall back to the
seed. A record from that fallback is evidence about the seed. A Linux ELF
selected on Darwin does not become native Darwin execution through resolution.

These distinctions matter for reproducibility. `--version` alone is insufficient
to identify source correspondence. The wrapper emits provenance on stderr and
can require a committed artifact with `SOUNIO_REQUIRE_COMMITTED_MADAROS=1`;
neither a version string nor that requirement means “rebuilt from this tree.”
Materialization of the compressed prebuilt verifies its tracked digest; it
does not rebuild modular source.

### Bootstrap and fixed points

`EXEC-004` — `source-observed`: `make build` compiles
`self-hosted/compiler/lean_single.sio` through successive generations and
compares seed generations. `build_modular_madaros.sh` instead derives or accepts
a source-tracking seed and uses it to compile `compiler/main.sio`. A seed fixed
point does not establish a Madaros fixed point. The modular build delegates
content caching and diagnostic receipts to `scripts/dev/madaros-cache.sh`;
an explicitly supplied seed remains part of the caller's provenance obligation.

The current tree also contains `madaros_fixed_point_gate.sh`. Its acceptance is
a ratcheted **rung**, not unconditional full fixed-point success: the default
expected rung is `run`, and the gate distinguishes checking, producing a second
generation, running it, and reproducing the executable payload in a later
generation. `.github/workflows/ci.yml` schedules that expensive rung for selected
post-merge, scheduled, or manually dispatched runs. A green job at a lower rung,
or a replay of a cached receipt, must not be described as a newly executed full
fixed point. No rung is claimed attained here.

The documented seed defect concerning nonfatal imported-module typecheck
errors remains a bootstrap limitation in `docs/compiler/BOOTSTRAP_SEED.md` and
the `CONVERGENCE FIX` logic in `lean_single.sio`. Its tolerance is not a Sounio
language permission to execute ill-typed code. This chapter neither changes
that policy nor infers that every present build exercises the defect.

## 6.3 Active pipelines and their boundaries

### The user compilation path

`EXEC-005` — `source-observed`: modular `check` reaches `run_check_mode` and
`compiler_visibility_preflight`. Despite the latter's name, it covers source
readability, module-closure parsing, and the multimodule typecheck, rather than
visibility alone. Success prints `check: OK`; it does not run the program or
establish that the requested backend can emit it. Optional `--verify-claims`
adds the claim executor at the checkpoint shown in the driver; it does not
turn an ordinary successful check into proof of every source assertion.

`EXEC-006` — `source-observed`: public native `build` and `compile` enter the
raw compiler's `build` mode, which calls `run_native_v2_compile_mode`, then
`module_frontend_compile_imported_to_file_for_target`. That routine scopes
target-dependent math, closure-environment lowering, and native Metal collection;
loads and checks the source closure; specializes and lowers bodies; resolves
call targets; and invokes `compile_native_v2_preview_to_file` on the resulting
`IrModule`. Despite the word `preview` in the writer's name, it is part of the
active compilation path. Reachability does not establish that any particular
artifact executes or that every opcode is implemented.

The single-module and specialized-collapse paths resolve local call targets
and ensure builtin targets; the multimodule path finalizes merged targets.
An optional cleanup pass is controlled by the raw compile mode's optimization
choice. The public launcher's `_parse_build_options` accepts a narrower option
set than the raw compiler: raw support for `-O` is not evidence that the same
flag is accepted by `bin/souc compile`. No source-level optimizer switch is
invented by this chapter.

`EXEC-007` — `source-observed`: the entrypoint and output API are part of a
pipeline claim. `compile_native_v2_preview_to_file` directly admits Linux x86-64
and the new Darwin arm64 writer and returns a refusal for other native targets.
The older `compile_native` result API can retry the legacy x86 emitter for
specifically classified unsupported-v2-opcode errors. Its Darwin arm64 branch
uses `compile_to_macho_arm64_preview`, a different implementation from
`compile_arm64_macos_to_file`. Raw positional modular invocation also passes
through additional legacy/thin-link dispatch. Therefore “the native backend
falls back” or “the native backend never falls back” is too broad without a
named caller. Neither statement licenses fallback in the public file route.

The `--probe-native-streaming` route is separately reachable. Its region gate
checks emitted region builtin bodies and refusal propagation, but expressly
does not execute the streaming output because the script records a prior
independent streaming execution failure. That comment is a historical defect
record, not a fresh reproduction. Compile coverage on this route cannot certify
ordinary user execution or repaired streaming execution.

### Intermediate representations

| Representation | Source and symbols | Current route and limits |
|---|---|---|
| Parser AST | `self-hosted/parser/ast.sio`, `self-hosted/compiler/module_parse.sio::load_module_file` | Source syntax and item trees used by checking and both CPU/GPU frontends. Parsing alone does not admit execution. |
| Checked artifacts | `self-hosted/check/mod.sio::check_program_with_artifacts`, `checker_apply_ir_metadata` | Type/effect/epistemic information attached or transferred during lowering; not a serialized portable machine ABI. |
| Native IR | `self-hosted/ir/ir.sio::IrModule`, `IrFunction`, `IrInstr`; `self-hosted/ir/lower.sio` | Active CPU path. Instruction storage uses arena regions and SoA accessors; an `IrFunction`'s compiler-storage region is not a program runtime region. |
| Native machine IR | `self-hosted/native/machine_ir.sio`, `self-hosted/native/lower_ir.sio` | Native lowering/model path. Core IR emission and machine-IR emission coexist; not every emitted instruction necessarily passes through the same register allocator. |
| HLIR | `self-hosted/hlir/lower.sio`, `self-hosted/gpu/hlir_to_gpu.sio` | Reachable GPU artifact frontend. A typed native XOR bridge also exists in `self-hosted/hlir/native_bridge.sio`; its existence does not make HLIR the universal CPU frontend. |
| ENIR and ENIR MIR | `self-hosted/enir/driver.sio`, `self-hosted/enir/source_lower.sio`, `self-hosted/enir/verify.sio`, `self-hosted/enir/interpreter.sio`, `self-hosted/enir/mir.sio` | Separately executable validation/interpreter/translation lane, not imported by the production native entry path inspected here. ENIR MIR and native machine IR are distinct. |
| SOIR binary formats | `self-hosted/ir/serialize.sio`, `self-hosted/ir/soir_reader.sio`, `self-hosted/ir/soir_v6_reader.sio` and writers | Serialization/roundtrip surfaces with version-dependent representability. Not a mandatory phase in public compile/run. |
| SOIR line format | `self-hosted/wasm/hlir2wasm_driver.sio`, `scripts/dev/hlir2soir.py` | Separate textual bridge to a WASM experiment; not interchangeable with the binary formats merely because both use “SOIR.” |

`EXEC-008` — `source-observed`: ENIR is a distinct, bounded lane. Its driver
checks canonical input, verifies modules, lowers selected source profiles,
interprets them, and can produce MIR-related receipts. `enir_driver_read`
accepts only its bounded input size, and execution records profile/fuel/stop
information where the schema supports it. The production compiler entry,
module frontend, and module native driver do not import `enir::`. Consequently
an ENIR verification, move/poison check, or numeric witness does not establish
that native IR or the public runtime enforces the same property. E1/E2/E3 gates
must be read with `madaros_v2_enir_gate_scope.sh`, which can classify protected
surface drift as a skip rather than execute the lane.

`EXEC-009` — `source-observed`: public raw `--ir-dump` and `--ir-roundtrip`
use the real binary serializer and a representability check.
`compiler_soir_unrepresentable` rejects metadata that the selected SOIR image
cannot retain, including named-return metadata, float-return metadata, selected
reference/wide-parameter fields, BSS/ontology state, and other enumerated fields.
The roundtrip compares bytes after decode/re-encode. It is a losslessness check
for admitted images, not an execution equivalence theorem or permission to
discard unsupported fields. Standalone v5/v6 arena readers have their own
positive, stale-region, mutation, and layout-refusal witnesses.

## 6.4 Sequential evaluation and observable storage

### Evaluation and control flow

`EXEC-010` — `source-observed`: the ordinary native expression lowerer
threads state from the left operand into the right operand, and its ordinary
argument-list lowerer traverses source arguments in order. Array-list literals
evaluate and store successive elements in order. For an inline callee expression,
`lower_call_expr_generic_ref` lowers the callee before its arguments. These are
descriptions of these routines, not a complete adopted language rule covering
every builtin, compound lvalue, kernel, or optimizer transformation.

`EXEC-011` — `source-observed`: native control flow is encoded by labels,
conditional branches, jumps, and returns. `lower_while_expr_ref` places the
condition on the loop path; `lower_return_expr_ref` routes eligible returns
through a region exit block. The kernel lowering also distinguishes a lane's
`return` from returning from the entire host launch. These distinctions are
observable whenever a condition, body, or return has I/O or mutation.

The ordinary `lower_binary_expr_ref` lowers both operands before emitting its
binary operation. No short-circuit branch is present there for `OpAnd` or
`OpOr`. This is observed lowering behavior and a language-contract gap, not a
normative declaration that Sounio must be eager. Short-circuiting remains an
open decision in `EXEC-040`. The available type-error fixtures do not test RHS
suppression. Metal's separate AST renderer spells `&&` and `||` in MSL; that
fact makes a cross-route side-effect witness necessary, rather than establishing
agreement. No runtime result for this discrepancy is claimed here.

`EXEC-012` — `source-observed`: module/global storage is represented using
BSS entries and initialization metadata. `module_frontend_global_init_compile_begin`
preserves initializer state across closure loading; the lowerer handles deferred
closed initializers, and the Darwin entry invokes `a64m_global_inits` before
calling `main`. A zero-filled BSS slot and a completed initializer are distinct
states. This chapter does not define observable ordering among arbitrary
effectful or cyclic module initializers: the implemented closed-initializer
mechanism is narrower than that language decision.

### Values and copies

`EXEC-013` — `source-observed`: a source aggregate value is not defined by
the identity of its backend handle. Native aggregate bindings can allocate and
copy rather than perform a single-word `IrCopy`. The lowerer has
`emit_ident_aggregate_value_copy`, `emit_fixed_array_value_copy`, and
`emit_struct_value_copy_budgeted`; the fixtures distinguish mutation of the
copy from mutation of the original. Nested struct fields and fixed-array
call boundaries have separate witnesses because a flat outer copy does not
establish deep independence.

There is an explicit residual limitation: `emit_struct_value_copy_budgeted`
falls back to `emit_struct_value_copy` when the computed deep-copy cost exceeds
available IR headroom. Its recursive copy routine also has a depth boundary.
This can retain shared nested aggregate handles. That source-observed fallback
must not become a normative exception saying “value copies alias when compilation
is expensive.” A recursive/deep/capacity discriminator remains necessary; this
chapter has not reproduced an alias at runtime. Independently, the existing
focused gate for fixed-array call boundaries witnesses direct word-scalar
arrays at its stated sizes. It does not cover the struct deep-copy fallback,
arbitrary recursive aggregates, dynamically shaped values, or foreign aggregates.

`EXEC-014` — `source-observed`: explicit references and raw object values
require different address operations. A reference can designate the slot holding
an aggregate representation; a field access or array-element borrow must reach
the intended pointee, not accidentally interpret the slot's address as an object
header. Lowering records ref metadata and uses distinct native ref/raw/managed
paths. `a64_array_element_borrow.sio` discriminates borrowing an array element
from slicing an array or mutating all repeated elements through one alias.

The static rejection of conflicting `&` and `&!` borrows is an independent
checker obligation, pinned by `borrow_call_explicit_release.sio` and
`borrow_call_conflict_explicit.sio`. Address generation does not prove borrow
soundness, stack-escape safety, or pointer validity after a raw free. Neither
the spelling `Panic` in an indexing function nor a handle table provides those
proofs.

### Arrays, strings, and raw data

`EXEC-015` — `source-observed`: array operations distinguish logical element
count, storage bytes, and representation. `ir_alloc_array` carries the logical
count; the active native `IrArrayLen` reads it from the object header's flags
field rather than dividing bucket capacity by an assumed element size.
`lower_index_expr_ref` classifies local/parameter aggregates, references to
arrays, raw word storage, packed unsigned bytes, and packed signed bytes.
In its current encoding, labels 0/1/2/3/4 distinguish these paths. Those labels
are backend metadata, not source type identifiers or a stable external ABI.

The native local/parameter/raw array load helpers inspected here perform
address calculation and a load without a logical-bound comparison. Darwin's
`a64m_array_index` is likewise an address calculation. Therefore a universal
claim of checked native fixed-array indexing is not supported. `Seq` access
and the Metal bridge have separate checks; their checks do not repair or
specify fixed-array access. Index `N`, a negative signed index, an empty array,
and a byte array with a smaller physical extent need dedicated refusal/trap
rules and witnesses. Raw access accidentally landing inside an overallocated
bucket is not successful bounds enforcement.

Repeat literals also have a representation-sensitive implementation:
`lower_array_repeat_fill_distinct` puts aggregate construction or copying
inside a runtime loop so each element receives a distinct object. Scalar
repeat filling can reuse a previously evaluated scalar word. Emitting one
initializer expression into the loop is different from evaluating it once
for the entire array. `madaros_array_repeat_aggregate_distinct.sio` tests
element independence; an effectful repeat initializer still needs an adopted
evaluation-count rule and a dedicated discriminator. An optimizer that hoists
the aggregate allocation out of this loop would destroy that independence.

`EXEC-016` — `source-observed`: strings on the inspected native paths use
addressable packed bytes and NUL-terminated runtime operations, rather than the
eight-byte-per-element array path. Darwin string literals embed a trailing
zero; string length scans bytes, and character indexing loads a byte.
Native string concatenation lowers to `str_concat`, not integer addition of
addresses. `str_from_bytes`, slices, and returned views must be considered with
their backing storage's lifetime; a pointer value is not an ownership transfer.

These facts do not specify Unicode scalar indexing, grapheme indexing,
normalization, handling of embedded NUL, or ownership of every slice. The
packed-string regression and a concatenation-as-call-argument regression are
useful discriminators, not a complete string language definition.

## 6.5 Runtime allocation and lifetime by target

### Linux x86-64 managed storage

`EXEC-017` — `target-contract`: the active Linux x86-64 native-v2 allocator
uses a runtime-context-backed bump heap and a bump handle table within a
2 GiB mapping. `native_v2_handle_table_capacity_default` is 4194304;
the handle entry is 48 bytes and the object header is 32 bytes. These are
implementation constants read from source, not computed resource requirements
or language limits. `nc_core_emit_alloc_into` distinguishes managed handles
from small-value allocations that return a raw object-base pointer without
consuming a handle entry. Such small allocations still consume heap space.

Managed object headers record descriptor, payload size, flags, and handle ID.
Array logical length is carried separately from allocation capacity. Handle
resolution can distinguish a handle index from a raw base under this runtime's
encoding. Neither is a portable C pointer contract. The compiler's own temporary
Box arenas, IR-storage arenas, explicit heap allocations, and program managed
heap are different lifetime domains even when their implementations use similar
bump/reset techniques.

Heap/handle pressure enters the allocation failure path, records diagnostic
context where provided, and can terminate with the handle-exhaustion status 182.
The handle-reclamation acceptance script deliberately treats this existing
baseline failure as failure to meet reclamation acceptance, rather than printing
a success for the repaired contract. No exhaustion or reclamation test was run
here.

`EXEC-018` — `target-contract`: Linux runtime region reclamation is an
eligible-function optimization with conservative escape retention, not a
source region type or general tracing collector. `lower_rgn_backend_opt_in`
activates it for the main compiler driver. A region records starting heap and
handle cursors. Eligible returns share an exit path. For supported aggregate
results, the lowerer can copy the young returned part, reset the dead allocation
window, and copy the survivor into retained storage. Scalar returns are also
checked for bits that could encode a young reference.

Stores into older storage trigger barriers; foreign calls and raw syscalls
conservatively mark open regions escaped. Reference-to-numeric casts need an
escape hook before a reversible encoding hides the address bits. An escaped
region retains allocations. Exhausting region-record capacity or declining an
ineligible return shape selects the no-region path; it does not authorize
reclaiming live values. `SOUNIO_NO_REGION_RECLAIM=1` disables this lowering.
ADR-010 records experimental status and additional ceilings; its old statement
that Madaros has only an x86 backend is superseded by the Darwin file writer.

`madaros_region_reclaim_gate.sh` contains mutation controls removing individual
field/global/index/raw-pointer/sequence/cast/syscall barriers. Those controls
are stronger evidence than an allocator counter alone, but remain unexecuted
here. The documented reservation/headroom boundary for region runtime symbols
also needs preservation when compiler capacities change.

`EXEC-019` — `source-observed`: runtime metadata is not an active collector.
The native contract advertises false for tracing GC, precise descriptor scanning,
handle relocation, pinning readiness, runtime retry, and current-frame root
scanning where those mechanisms are unwired. Runtime pin fields use an unwired
sentinel. Safepoint or stack-map records do not establish a precise live-root
bitmap. Region reset is compatible with these limits; it does not turn them
into true collector capabilities. `gc_capability_honesty_gate.sh` and
`precise_stack_maps_honesty_gate.sh` guard this distinction structurally.

### Explicit allocation and stack frames

`EXEC-020` — `target-contract`: Linux `heap_alloc`/`heap_free` are a separate
raw mapping ABI. `emit_builtin_heap_alloc_into` rejects nonpositive requests,
checks request arithmetic, maps zeroed memory, stores the mapping length before
the payload, and returns a null result on allocation failure. `heap_free(NULL)`
is a no-op; a nonnull pointer must come from the corresponding allocator.
Managed handles and interior borrowed pointers must not be treated as such
raw allocations. The POSIX `malloc/free` compatibility path and arbitrary
dynlinked C allocation must also be identified by their actual bridge; a shared
name does not establish identical allocation ownership.

Native virtual registers commonly occupy function-frame slots. Linux and Darwin
have distinct frame/argument emitters, and compiler stack reservation is separate
from the storage lifetime of a user's locals. `MADAROS_STACK_KB` configures the
launcher, while the Darwin bootstrap's mapped-stack entry has its own guard
page. A large compiler stack reservation does not make an escaping address to a
dead frame valid.

### Darwin arm64 storage

`EXEC-021` — `target-contract`: `codegen_arm64_macos.sio` emits object
allocations with a header and process-lifetime mmap storage. It does not implement
Linux's reusable managed regions or tracing collector. The recognized
`__rgn_enter` returns zero, the lowering protocol's explicit no-region token;
`__rgn_escaped` is conservative, and a nonzero-token scalar close terminates
instead of silently ignoring an actual region request. An allocation failure
can trap. The Linux handle-exhaustion status must not be advertised as Darwin's
allocation-failure contract.

Escaping closures on this target can retain heap environments with process
lifetime. This solves a particular frame-lifetime problem; it does not establish
bounded memory growth, reclamation, reference counting, or cross-target closure
equivalence. PR2978's current guide describes this lifetime explicitly.

## 6.6 Internal calls, function values, and foreign boundaries

The ABI rules here describe the inspected emitters. They do not promise a stable
binary interface across compiler revisions. `native/abi.sio`,
`native/abi_lower.sio`, and `native/ffi.sio` contain useful ABI infrastructure,
but their declarations must not replace inspection of the reachable emitter.
In particular, an ABI table with AAPCS/Win64 register lists is not proof that the
public target writer implements that table or can call arbitrary C functions.

`ABI-001` — `target-contract`: ordinary Linux x86-64 internal calls transport
the first six word arguments in `rdi`, `rsi`, `rdx`, `rcx`, `r8`, and `r9`, with
additional arguments in the emitter's aligned outgoing stack area. The result
travels through `rax`. Internal floating scalar values travel as word bits;
float metadata determines how arithmetic and later uses interpret them.
Consequently this is a Sounio internal convention using SysV integer rails,
not a claim that a source `f64` parameter uses the external C floating ABI.

`ABI-002` — `target-contract`: the Darwin file emitter uses `x0` through
`x7` for the first eight internal word arguments, its own stack argument layout
for the rest, and `x0` for a scalar result. `a64m_emit_function` spills parameter
words into virtual-register slots. Scalar `f64` bits travel on the internal
integer rails and are moved to floating registers for arithmetic or external
math as required. Parameter and virtual-register ceilings are checked by the
emitter. This is not an assertion of general AAPCS aggregate interoperability.

`ABI-003` — `source-observed`: aggregate parameter/return representation must
be derived from lowering metadata and the selected emitter, rather than a source
size rule borrowed from C. Ordinary native aggregate calls can exchange a
handle or base pointer, with value copying performed at separate lowering sites.
`return_struct_name`, fixed-array metadata, ref masks, and float metadata inform
those sites. The small-aggregate allocator threshold is not a universal rule
that source structs of that size return as two C ABI registers. Array-containing
returns, imported calls, and explicit out parameters have distinct fixtures in
`tests/selfhost/native_runtime/abi_manifest.tsv`.

`ABI-004` — `source-observed`: explicit sret IR is a separate representation.
`IrCallSret` carries a caller-provided destination, explicit argument count, and
the contiguous first explicit argument register in `field_idx`; `is_sret` and
`sret_dest_reg` tell the callee how to receive the hidden destination.
The Linux emitter injects it in `rdi` and shifts explicit arguments; Darwin
injects it in `x0` and shifts arguments through `x1` onward. `IrReturnSret`
returns through the matching convention. This hidden `x0` convention must not
be replaced in documentation with an assumed external AAPCS `x8` rule.

The inspected explicit sret producers include compiler-owned IR witnesses.
Finding those producers and emitters does not show that every user aggregate
return is transformed into sret. The raw `--native-v2-emit-sret` witness and
ordinary source struct-return fixtures test different boundaries. A portable
rule choosing direct versus sret representation remains open.

`ABI-005` — `source-observed`: direct and indirect calls preserve different
identity information. Direct calls use function IDs and relocations; a function
value uses `IrLoadFnRef` and `IrCallIndirect`. The frontend resolves merged
targets before emission. Empty or unresolved required bodies must be diagnosed
or refused, not replaced by an arbitrary zero result. Darwin computes a live
function set from `main`, direct calls, sret calls, and function-address loads,
then refuses unresolved emitted calls. Reachability elimination does not make
an unrelated, uncalled declaration executable.

`ABI-006` — `target-contract`: zero-capture function references, locally
lowered captured calls, and escaping closure values are distinct cases.
The target-scoped `lower_swap_native_closure_envs` enables Darwin's heap
environment representation; an odd function value denotes a closure descriptor,
with a hidden environment argument and a code pointer. The emitter tests the
tag before indirect dispatch. Linux/default restrictions on opaque or escaping
captured closure uses must not be erased by citing Darwin's positive witness.
Capture capacity refusal and inferred parameter types have separate negative
fixtures. Heap environment transport does not override linear-capture checking.

`ABI-007` — `target-contract`: Linux FFI admission is the combination of the
checker's supported-extern policy, builtin recognition, any supported dynamic
link stub, and the call-site marshaling actually reached. `E250` refuses an
unimplemented extern call; an undeclared ordinary call has a different diagnostic
(`E137` in the gate's control). A declaration never called can remain legal.
The POSIX gate pairs these controls with real PID identity, raw allocation
write/read, exit-7, abort, and `system` side-effect/wait-status witnesses.

The dynamic-link stub in `emit_dynlink_got_stub_into` preserves open-region
lifetime conservatively. The larger `FfiContext`, callback, Win64, and marshaling
model in `native/ffi.sio` is not evidence of arbitrary callable C ABI support.
Variadics, callbacks, retention by foreign code, and complex aggregate signatures
need their own reachable marshaling and runtime witnesses.

`ABI-008` — `target-contract`: Darwin external calls are narrower than its
internal calls. `a64m_system_math` maps supported math names through libSystem
imports, with integer-bit/internal to floating/external conversion. The file
writer has a dedicated Metal import bridge. An unrecognized arbitrary external
symbol is not promised; it must not be reported as a usable foreign function
merely because the parser accepted an extern declaration. The scoped Darwin
math checker setting is a target capability, not relaxation of all check modes
or of the Linux extern policy.

`ABI-009` — `target-contract`: `syscall6` is an OS-specific low-level escape
boundary. The Linux emitter shuffles the internal argument tuple into the Linux
syscall registers, including kernel argument four in `r10` and the sixth argument
from the incoming stack slot. It conservatively marks regions escaped before
calling the kernel. The raw compiler's run helper explicitly chooses Darwin or
Linux fork/wait/exit/chmod numbers. A source numeric syscall ID does not name a
portable operation across operating systems, and a syscall return word is not
the same thing as a high-level `Result` or a compiler exit status.

## 6.7 Failure, diagnostics, and process results

`EXEC-022` — `target-contract`: the inspected native `panic` builtin is
nonreturning process termination with status 1, and a false native `assert`
terminates as well. Linux's `emit_builtin_panic_into` does not print its argument
or unwind frames; the Darwin recognized builtin also emits an exit. No language
guarantee of panic message output, stack backtraces, destructor execution,
catching, region cleanup on unwind, or exception recovery follows from these
emitters. Compiler drivers frequently print specific diagnostics before calling
`panic`; that earlier print is a separate action.

Raw arithmetic traps, allocation failure, explicit foreign `abort`, checker
refusal, and normal user return are different termination classes. In particular,
the Darwin wide-IR gate distinguishes an invalid-op emission refusal from an
executed divide-by-zero trap; neither is a source type-error diagnostic. The
POSIX abort witness expects a signal-derived status distinct from `exit(1)`.

`EXEC-023` — `source-observed`: a meaningful refusal test needs diagnostic
attribution and output-artifact checks as well as a nonzero process status.
`scripts/lib/classify_compile.sh` separates accepted, refused, undiagnosed
failure, refusal-with-artifact, silent failure, crash, timeout, and infrastructure
failure. Its classifier treats timeout/kill conditions differently from an
ordinary diagnostic refusal. A gate that merely accepts `nonzero` provides
weaker evidence and must be described accordingly.

Diagnostics are emitted through several subsystems and formats. The modular
checker uses forms such as `error[E...]`; the seed has additional text shapes;
target and lowering refusals can use named markers such as `ARM64_REFUSAL` or
`HLIR_LOWERING_REFUSED`. `diagnostics/mod.sio::Diagnostic` does not establish a
single live machine-readable protocol for all of them. The raw driver prints
banners and progress to stdout outside explicitly handled protocol modes such
as `--serve`; stdout from raw compile/run is not necessarily program-only text.

`EXEC-024` — `source-observed`: compiler results and OS results are separate
types of evidence. `CompileResult` reports success/errors/output;
`NativeCompileResult` reports backend status, error code/message, output format,
and byte buffers. The large-file writer returns its own status codes instead.
An invalid target, unsupported opcode, capacity overflow, failure to write,
and a user program returning 7 must not be collapsed into “result = 7.”

The public launcher checks compiler status and a nonempty artifact, removes
partial output after failed native or GPU compilation, and adds an executable
mode for native output. Raw backend writers do not all offer the same atomicity
or stale-file cleanup: a refusal leaving a pre-existing output path unchanged
does not certify that old file. `native_capacity_tiers_gate.sh` is a structural
capacity-coherence/refusal gate, not fresh execution at every declared ceiling.
Its distinct internal failure codes are not a stable language-level error enum.

`EXEC-025` — `source-observed`: public `run` compiles into a unique temporary
directory, signs Mach-O output on Darwin, executes it with supplied arguments,
captures its process status, removes temporary output, and returns that status.
The launcher bounds compiler work, but `_run_source` itself does not add a
runtime timeout to the resulting program. A harness must add one when required.
Compiler chatter can precede program output. Normal nonzero user exit is not
evidence that compilation failed.

`EXEC-026` — `source-observed`: raw Madaros `run` has a separate spawn ABI
from public `bin/souc run` through the shell launcher.
`native_v2_run_elf_and_exit` sets mode, signs on Darwin, builds a bounded argv
array using arguments after `--`, spawns, waits, converts wait status, and exits
with the child status. It supplies an empty environment vector and has a finite
argument capacity. Public shell-wrapper execution inherits its shell environment
instead. These are observable differences, not interchangeable demonstrations
of one fully specified run-environment contract.

Payload-bearing source `Result`/enum behavior is not specified by either
compiler result struct. The current known-limitations ledger records general
payload-enum runtime representation as an open area, while specialized
Option/Box and compiler-internal patterns have narrower lowering. Source result
values are ordinary values where implemented; they are not automatically
exceptions, panics, or shell exit codes.

## 6.8 Target capability and refusal matrix

`EXEC-027` — `target-contract`: Linux x86-64 has the active native-v2 file
writer and ELF route. A claim about actual output still requires identifying
the compiler, module source, optimization flags, runtime parameters, and artifact.
The presence of SIMD emitters does not guarantee the executing CPU supports all
instructions selected by a particular scientific operator. Native register-policy
tables and feature metadata are not general CPU admission receipts.

`EXEC-028` — `target-contract`: PR2978's current-source Darwin arm64 route
is integrated in source and live scripts. `build_madaros_macos_arm64.sh` requires
a signed current-source native seed and `timeout`, builds a scratch candidate,
signs it, runs bounded acceptance, and requires additional wide-IR, port, and
pinned-corpus/Metal gates before installing the canonical host artifact. An
explicit noncanonical scratch output runs the bounded acceptance stage only.
The file backend emits Mach-O with dyld/libSystem integration. Raw `build`
does not sign its output; raw/public run or an executing harness performs signing.

The guide and corpus report describe historical original/corrected fixtures,
multiple candidate hashes, a pinned membership manifest, real Metal observations,
and a later main-integration capacity regression. These are different receipts.
The current `.github/workflows/ci.yml` includes native build, ABI acceptance,
positive/negative port regressions, wide-IR and Metal-IR execution gates, and the
corrected 200-member corpus. Earlier attest-only steps in the same job do not
turn later execution steps into attestations, and source workflow wiring is not
proof that a particular run passed. This chapter re-executes none of those gates.

`EXEC-029` — `target-contract`: target recognition and register policy do
not establish an output writer. Although `native_target_parse` recognizes more
targets and `target_policy.sio` has emitter-ready entries, the public file writer
rejects native targets other than Linux x86-64 and Darwin arm64. The older result
API separately reports stripped Mach-O/ARM64/PE routes and a RISC-V contract-only
preview. Windows x64, Windows arm64, Linux arm64, Darwin x64, and RISC-V therefore
cannot be promoted from parser names, architecture encoders, PE/Mach-O structs,
or model tests alone.

`EXEC-030` — `source-observed`: no public general WASM source-to-runtime
contract is established by this tree's imports or help text. The compiler's
WASM probe logs a dispatch marker; standalone `wasm/driver.sio`,
`wasm/hlir2wasm_driver.sio`, and lower/emitter modules provide separate bounded
experiments. The native `build --target wasm` file route is not a general WASM
writer. `scripts/build/build_playground_wasm.sh` explicitly creates a browser
JavaScript preview shim and an empty valid WASM module, and says no Sounio
compilation occurred. This artifact cannot certify compiled Sounio execution.

`EXEC-031` — `source-observed`: the raw positional modular compiler recognizes
`llvm`, but `compile_multimodule_llvm` prints that the backend is unavailable and
returns failure. Neither help text nor a repository LLVM-related component
supplies the missing public route. This is an explicit refusal boundary.

`EXEC-032` — `source-observed`: public `build --backend gpu` requests a GPU
artifact, defaulting to `cuda-sm80`, and dispatches through
`run_gpu_compile_pipeline`, checked AST/HLIR, and target-specific renderers.
Unsupported HLIR operations, user calls, compound shapes, or unsupported literal
forms are refused before publication where guarded by that pipeline. PTX text,
MSL text, SPIR-V bytes, and a cubin are different artifacts with different
validation needs. An emitted artifact is not a launched kernel. The current
SPIR-V branch uses `hlir_kernels_to_spirv_artifact`; the nearby old comment
describing a PTX placeholder must not override the actual branch.

Some GPU source paths also provide CPU lane-loop lowering. A successful CPU
execution of a `kernel fn` fixture is not by itself a CUDA or Metal launch
receipt. Conversely, the native Darwin Metal bridge does not use the entire
public HLIR artifact pipeline: it collects a target-scoped AST subset and emits
runtime-launch calls. These routes require distinct witnesses and refusal cases.

`EXEC-033` — `target-contract`: native Darwin Metal launch is real runtime
work through `gpu/native_metal.sio`, `a64m_metal_launch`, and
`gpu/runtime/native_metal_bridge.mm`. The bridge compiles MSL on an actual Metal
device, validates the supported parameter kinds and array extents, copies array
payloads to buffers, submits work, waits for completion, and copies writable
buffers back. Aliased array arguments are recognized by header identity and
share the bridge buffer. Device, source, pipeline, command, argument, bounds,
or execution failures return an error; there is no CPU fallback in this bridge.

Source `f64` uses eight-byte software binary64 values with pinned
`vendor/metal-softfloat/softfloat64.metal` arithmetic. This is neither native GPU
binary64 nor silent conversion to f32. An execution trace marker records a real
launch for this route, but the binary64 conformance gate and workload output
checks are needed to assess arithmetic. The supported AST/parameter subset is
narrower than all Sounio expressions. Zero lanes return before device work;
that successful no-op must not be counted as a real launch.

`EXEC-034` — `target-contract`: GPU launch width, indexing, and return have
route-specific execution meaning. Native Metal validates buffer length against
lanes and uses `sounio_tid`; native CPU kernel lowering has a lane-continuation
label so an early lane return does not abandon remaining lanes. A PTX artifact
must be tested by an actual CUDA launch harness, including transfer, lane count,
output, and error observation. No cross-target memory-order, barrier, race,
subgroup, or atomic semantics are established merely by this shared source
syntax.

`EXEC-035` — `source-observed`: Pireus is not another OS executable format.
Its reachable compiler surfaces select typed algebraic operations and target
recipes, while ontology/profile/admission tools describe and admit candidates.
The typed XOR contract carries `operator_kind`, bit width, twist, candidate,
and arity; semantic selection is not a callee-name magic string. The native
bridge `hlir_xor_convolution_to_native_ir` refuses nonmatching metadata. GPU
materializers have distinct DGX/PTX and Metal paths.

`pireus_typed_xor_gpu_gate.sh` requires equivalent renamed source operators,
rejects magic-name poison and f32/f64 confusion, and rejects extra-store,
reordered-operand, and duplicate-operator shapes with target-specific refusal
markers. Its positive artifact pattern/hash checks do not execute the target
instructions. `pireus_admission_engine_gate.sh` executes a separate Sounio
admission tool and attacks it with an always-admit control; admission does not
establish floating-point parity, a general compiler backend, or hardware speed.
Historical Pireus engine/ontology observations are not current device inventory.

| Target/route | Source admission and writer | Positive discriminator | Refusal/limit discriminator | Claim boundary |
|---|---|---|---|---|
| Linux x86-64 public native | `compile_native_v2_preview_to_file` / ELF writer | `tests/selfhost/native_runtime/import_core_abi_42.sio` | `madaros_region_reclaim_gate.sh` forced emitter refusal; native capacity gates | Active source route; no fresh artifact here. |
| Darwin arm64 public native | `compile_arm64_macos_to_file` | `a64_unit_literal_return.sio`, exit-status fixture | `madaros_macos_port_regressions_gate.sh`, unresolved calls/opcodes | Current-source Mach-O route, separate from Linux parity and self-hosting. |
| Windows x64/arm64 | Target parser/policy only on public file path | No public executable witness established here | File writer rejects target; result API reports stripped PE path | No public Windows runtime promotion. |
| Linux arm64 / Darwin x64 / RISC-V | Parser/model/encoder surfaces | Target-scoped model tests only where present | Public file writer refuses; older result API distinguishes stripped/contract-only | Names and policy entries are insufficient. |
| WASM / browser | Standalone WASM experiments and browser shim | `tests/wasm/add.sio` in its dedicated lane | No general native-build WASM writer; empty preview module | Experimental lane; browser preview is not compiler output. |
| LLVM | `compile_multimodule_llvm` | No successful route claimed | Explicit unavailable message and failure | Refused in this modular route. |
| CUDA/PTX public GPU | Checked AST → HLIR → PTX/other requested artifact | Source-level GPU fixtures and CUDA gates | `gpu_lowering_refused`, precision/compound-shape controls | Artifact validity and execution are separate. |
| Native Darwin Metal | AST subset → native launch call → dylib/MSL | Metal-IR gate and mandatory-Metal corpus | Runtime errors, unsupported AST/ABI, binary64 conformance | Real device bridge in source; no launch performed here. |
| Public GPU Metal / SPIR-V | HLIR target artifact writers | Artifact gates | Literal/shape refusals as implemented | Not the native Darwin runtime bridge or general backend parity. |
| Pireus | Typed operator metadata plus separate admission/profile tools | Renamed sedenion operator fixture | Magic-name poison, f32 mismatch, compound-shape refusal | Typed materialization/admission, not blanket runtime or performance evidence. |

Gate and fixture basenames in this matrix are expanded to exact repository
paths in the evidence catalog and the conformance table below.

## 6.9 Optimizer legality and scientific observability

`EXEC-036` — `specified-contract`: preserve scientific meaning across
lowering and optimization. `FOUNDER_INTENT.md` explicitly requires preservation
of order and parenthesization for nonassociative computations, prohibits silent
precision demotion, and distinguishes computed value, numerical error,
measurement uncertainty, provenance, and confidence. Thus reassociating an
octonion/sedenion expression or substituting f32 for f64 cannot be justified
merely by selecting an available target instruction. This is an existing project
requirement; this chapter supplies no new proof that every pass satisfies it.

`EXEC-037` — `source-observed`: the active cleanup pass has named legality
checks and distinct authority domains. `ocp_rebracket_is_exact_bitwise_op`
admits only AND/OR/XOR for its exact-bitwise rebracketing transaction, with
checks on occurrence identity, use modeling, float markers, control flow,
dominance, and intervening writes. Ontology parameter links are audited after
the authority-owned decision; an unrelated scientific observation cannot issue
compiler rewrite authority.

The separate epistemic e-graph pass is gated by precision-preserving strategy,
observed registers, and algebra metadata. Its chaotic variant refuses
reassociation, while algebra-specific strategies include selective and blocked
cases. This is a description of guards, not an endorsement of every candidate
rewrite, a universal bitwise-reproducibility theorem, or a claim that every
algebraic reassociation is safe. The cleanup sequence also contains CSE, GVN,
copy propagation, dead-store elimination, LICM, and SCCP; checking one rebracket
authority does not prove legality of those other passes.

For testing optimizer legality, compare the same source and compiler with the
actual optimization choices for that route, observe all required channels,
and include special values, aliasing, calls, mutation, and expression-tree
differences. `madaros_ieee_f64_special_values_gate.sh` addresses NaN comparisons
and special-value printing, not every rounding mode. Exact C++23 oracles used
by the Darwin wide-IR and Metal binary64 gates are scoped numerical checks;
neither a matching value nor a source-level Lean file proves the whole compiler.
No proof assistant build was run for this chapter.

## 6.10 Evidence and conformance obligations

`EXEC-038` — `specified-contract`: an execution claim must be accompanied by
a receipt adequate to identify and reproduce the claimed path. At minimum it
names source revision and dirty state, source/fixture identity, selected compiler
engine and executable identity, relevant resolver overrides, target, options,
stdlib/runtime dependencies, exact command, artifact identity/format, execution
host, stdout/stderr, exit or signal outcome, and the oracle/assertions used.
Signature state is relevant for Darwin. GPU claims additionally identify device,
driver/runtime, actual launched artifact, transfer/launch completion, and whether
a fallback occurred. An artifact hash without execution output is custody
evidence, not execution evidence.

`EXEC-039` — `source-observed`: the serious-language conformance spine is a
bounded set of check/run/refusal cases in `tests/conformance/manifest.v1.tsv`.
Its compiler defaults to this checkout's `bin/souc`; that default still needs
engine/artifact resolution. Its `compile-fail` mode invokes `check`, and its
`nonzero` predicate alone does not classify crashes/timeouts. The companion
spec-drift gate validates catalog relationships and selected results, while
the claim-closure gate governs public claim wording. None is a universal
execution, ABI, target, or optimizer proof.

The table states what each existing test can discriminate, not a result from
this specification session. A negative cell can be a mutation/refutation gate,
an adversarial runtime assertion, or a compile-fail fixture; those are different
kinds of negative evidence.

| Boundary | Positive source or gate | Negative/control source or gate | Required interpretation |
|---|---|---|---|
| Compiler override identity | `scripts/ci/compiler_override_fail_closed_gate.sh` | Same gate's unusable overrides | Launcher selection/refusal, not compiler semantic execution. |
| Public failure propagation | `scripts/ci/madaros_launcher_exit_status_gate.sh` | Generated malformed source and partial artifact checks in that gate | Error attribution and artifact absence, not merely nonzero. |
| Darwin host and ordinary output | `scripts/ci/madaros_macos_arm64_acceptance_gate.sh` | `tests/native/madaros_macos_exit_status.sio` returns 7 | Correct process status distinguishes successful compilation from user failure. |
| Full Darwin selection | `scripts/ci/madaros_macos_corpus_gate.sh` | `scripts/ci/madaros_macos_corpus_diagnostics_gate.sh` | Pin all manifest members and fixture snapshots; invalid-compiler control tests the harness. |
| Borrow/address interpretation | `tests/run-pass/a64_array_element_borrow.sio` | `tests/compile-fail/borrow_call_conflict_explicit.sio` | Runtime pointee identity and checker exclusivity are separate obligations. |
| Local aggregate values | `tests/run-pass/aggregate_local_binding_is_value_copy.sio` | Assertions in `tests/run-pass/aggregate_nested_field_deep_copy.sio` | Outer scalar-copy success alone does not cover nested field independence. |
| Word-array copies | `tests/native-v2/local_known_extent_word_array_copy_witness.sio` | `tests/known_failures/madaros_fixed_array_call_boundary_alias_probe.sio` | The historical directory does not decide current outcome; gate compares expected behavior. |
| Fixed-array parameter isolation | `tests/native-v2/fixed_array_call_boundary_value_witness.sio` | `scripts/ci/madaros_fixed_array_call_boundary_alias_gate.sh` | Direct arrays and mutable-reference aliasing must be distinguished. |
| Struct/ref/out/return ABI | `tests/selfhost/native_runtime/abi_manifest.tsv` | `scripts/ci/native_v2_imported_hof_abi_gate.sh` boundary checks | Explicit expected exit 42 is a success for these fixtures, not default exit-zero testing. |
| Region safety | `tests/run-pass/madaros_region_reclaim_escape.sio` | `scripts/ci/madaros_region_reclaim_gate.sh` barrier sabotage | Default runtime execution differs from streaming body-emission inspection. |
| Managed reclamation | `tests/native-v2/handle_reclamation_below_capacity.sio` | `tests/native-v2/handle_reclamation_exhaustion.sio` | Current failure 182 cannot be counted as successful reclamation acceptance. |
| FFI truthfulness | `tests/ffi_posix/arm_control_implemented.sio` | `tests/ffi_posix/arm_claim_unimplemented.sio` | Implemented behavior, undeclared-call specificity, and unimplemented refusal all matter. |
| Packed string indexing | `tests/data_io_gated/string_index_packed.sio` | `scripts/native_string_index_packed_gate.sh` checks | Byte representation and supplied I/O path; no Unicode indexing promotion. |
| Escaping Darwin closure | `tests/run-pass/madaros_escaping_float_closure.sio` | `tests/compile-fail/madaros_closure_capture_capacity_refuses.sio` | Target-scoped lifetime/return typing and clean capacity refusal. |
| Binary serialization | `tests/fixtures/soir/v6_calls_and_lets.sio` | `tests/fixtures/soir/globals_extern_call.sio` | Representability/refusal and byte roundtrip; no runtime promotion. |
| Standalone SOIR layout | `tests/native-v2/soir_v6_bss_layout_witness.sio` | `tests/native-v2/soir_v6_bss_layout_reject_witness.sio` | Separate reader/writer contract from the public CLI's admitted subset. |
| ENIR memory/moves | `scripts/dev/madaros_v2_e2h_enir_memory_move_poison_gate.sh` | Same gate's move/poison controls | Bounded ENIR lane; inspect scope-skip before reading a green process exit. |
| Darwin wide arithmetic | `scripts/ci/madaros_macos_wide_ir_gate.sh` | Generated invalid operation and zero-divisor controls | C++23 oracle → emitted IR → signed program, distinct from source-wide ABI coverage. |
| Metal binary64 | `scripts/ci/metal_binary64_conformance.cpp` | Its special-value/arithmetic checks and bridge errors | Tests pinned software binary64 on a device; not native GPU f64. |
| Pireus typed selection | `tests/gpu/sedenion_mul_source_level_renamed.sio` | `tests/gpu/pireus_magic_name_poison.sio` | Rename independence and poison nonselection, not instruction execution. |
| Pireus terminal shape | `tests/gpu/sedenion_mul_source_level.sio` | `tests/gpu/pireus_metal_extra_store.sio` | Compound-shape refusal must be for the intended reason. |
| Exact-bitwise rewrite | `tests/compiler/rebracket_authority_cross_block_source.sio` | `tests/compiler/rebracket_authority_nondominating_source.sio` | Refusal of an optimization is distinct from refusal of the program. |
| Optimizer authority | `scripts/ci/exact_bitwise_rebracket_source_ir_gate.sh` | `tests/compile-fail/rebracket_authority_unrelated_ontology_obligation.sio` | Default source-check-only mode is weaker than strict executable mode. |

The full serious-language gates and many target gates can build compilers or
run legacy tooling. They were read, not invoked. Their presence here is not a
request to bypass the authorized no-heavy-build boundary or ADR009.

## 6.11 Open decisions and known gaps

`EXEC-040` — `open-decision`: adopt a language-wide evaluation-order and
logical-operator rule. The evidence supports ordinary native left-to-right
lowering, but does not settle short-circuiting, every compound lvalue's
single-evaluation behavior, builtin special cases, or kernel side-effect order.
The required discriminator is an effectful RHS/callee/index counter observed
under each relevant route and optimization setting. Typechecking a bool
expression does not answer this question. Eager emission in one routine must
not silently decide the language.

`EXEC-041` — `open-decision`: complete the memory-safety and lifetime contract.
The unresolved scope includes fixed-array bounds behavior, invalid/raw pointers,
stack-reference escape, deep-copy budget/depth fallbacks, view/string ownership,
cyclic initialization, allocator exhaustion, and whether any portable reclamation
guarantee is promised. Candidate designs include explicit refusal, defined traps,
or a checked safe surface with separately marked raw operations; this chapter
does not adopt one. A target's current unchecked load is a coverage/implementation
gap, not a declaration of language-level undefined behavior.

`ABI-010` — `open-decision`: decide whether a stable external Sounio ABI is
intended, and define direct/sret thresholds, aggregate layout, parameter/result
classification, function-value representation, capture ownership, C variadics,
callbacks, foreign retention, and unwind interaction per supported target.
Current internal word rails, handles, and heap descriptors are implementation
facts. Neither the generic ABI model nor a six-function imported fixture creates
a portable public ABI. ABI versioning and compatibility tests remain necessary
before distributing independently compiled binaries as interoperable units.

`EXEC-042` — `open-decision`: define the portable failure/result interface.
The missing decisions include panic diagnostics and cleanup, language error
values versus process status, general payload-enum lowering, the run environment,
signal mapping, and a versioned machine-readable diagnostic protocol. A candidate
contract should keep semantic refusal, internal compiler failure, resource limit,
target unavailability, user panic, foreign abort, and ordinary nonzero user return
distinguishable. No single universal error code map is adopted here.

`EXEC-043` — `open-decision`: define concurrent and accelerator memory
semantics before claiming cross-target equivalence. This includes races,
atomics, barriers, device/host visibility, asynchronous launch lifetime,
reproducible reductions, fusing versus nonfusing arithmetic, and partial device
failure. The synchronous native Metal buffer bridge provides a narrow concrete
completion contract, not a language memory model for all GPU, Async, or foreign
execution. General CPU threading/async runtime behavior was not established by
this chapter's native-route research.

Known defects and limitations localized by this research are the source-level
deep-to-flat copy fallback, unchecked native indexing helpers, the ordinary
binary lowerer's eager operand emission, explicit unsupported target routes,
and route-specific limitations of raw run and closure transport. They are
recorded for specification accuracy. Compiler repair is outside this chapter's
authorized write set; no new issue, ownership registry, or blocker closure is
created by describing them.

## 6.12 Research and validation record

Inspection covered the public launchers and shared resolvers; bootstrap and
fixed-point scripts; compiler main/module frontend/native drivers; native
Linux and Darwin emitters; GC/runtime context; ABI and FFI models; IR lowering,
copying, serialization, and optimizer guards; ENIR and HLIR boundaries; the
native Metal bridge; PR2978 guides and acceptance/corpus scripts; Pireus typed
materialization/admission gates; and serious-language claim/conformance policy.
Symbol names and repository-relative paths are used instead of guessed source
line numbers.

Two Darwin `coder-next` consultations supplied a generic checklist and a draft
editorial review, without repository access. Narrow clarifications about active
compilation versus execution, open evaluation rules, copy-fallback coverage,
and raw/public run distinctions were incorporated. Unsupported suggestions were
not treated as implementation facts or semantic authority. All implementation
statements above are based on local source inspection. Shared governance
registration, central specification indexing, and shared offload logging remain
with the parent coordination lane.

Only document/catalog consistency checks belong to this chapter's validation
record. No test listed in the tables is represented as having passed during
this work; no historical hash or corpus count is asserted to describe a current
installed artifact. Legacy entrypoints and alternate pipelines are documented
and remain untouched.
