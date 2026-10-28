# Array representation and cast boundary: source and native observations

This receipt explains why the library's typed indexing repair is separate from the proposed checker guard. It does not establish array-to-element-pointer decay, repair element borrows, or accept a new integrated compiler.

## Source contract

The exact source-observed TYPE-017 text at docs/spec/2026-10/02-types-generics-and-ownership.md:333 is:

> A successful annotation check is not a license to obtain another representation by `as`; runtime bit reinterpretation and numeric conversion must be specified operation by operation.

EXEC-014 at docs/spec/2026-10/06-execution-memory-and-targets.md:270 states:

> A reference can designate the slot holding an aggregate representation; a field access or array-element borrow must reach the intended pointee, not accidentally interpret the slot's address as an object header.

These statements do not promise payload decay for an array value or an array reference. A raw pointer to the same array type can preserve its slot representation. An element pointer requires an actual element-address operation; replacing a type by `as` does not supply one.

## Frozen implementation and IR

Raw compiler SHA256 is 11332ed8a6cf7ccb85f0a31bf63bed1fcc10659054d68639d6dba233a6014ff4. The frozen lower.sio, ir.sio, codegen_x86_linux.sio and check.sio used for this investigation are included in array-semantics10/input/frozen.

The generic fallback in lower_cast_expr_ref returns the original source register after its specific supported conversions. It does not insert payload decoding for an array cast. emit_builtin_read_offset64_into emits the raw word load MOV RAX,[RDI+RSI*8];RET (48 8b 04 f7 c3), shared by read_i64 and read_f64 under the word ABI. This primitive must remain a raw load. OpRef/OpRefMut materialize the address of the slot. Typed indexing instead resolves the array representation through the existing managed-array access path.

The NV2 trace is enabled explicitly by SOUNIO_NV2_IR_TRACE=1. It shows the array allocation result reaching read_f64 unchanged for value4, and a reference register reaching read_f64 for ref4. The tracer labels the unary reference instruction as "other"; its interpretation as LEA comes from the frozen lowering/codegen and the source, not from an opcode name invented by the trace reader.

## Slurm 19448

The fifteen nonempty source producers run in both SOUNIO_LOOP_REGIONS=0 and 1. All thirty check and compile phases succeed. Native and independent C++23 bit-oracle outcomes remain separate:

| Source category | Runs | Native result | C++ oracle |
| --- | ---: | --- | --- |
| Typed local array indexing | 4 | 0 | pass |
| Array value cast to element pointer | 6 | SIGSEGV 139 | refusal |
| Array reference cast to element pointer | 6 | 0, observed bits 1 | refusal |
| Heap read_f64, read_i64 and raw indexing | 6 | 0 | pass |
| Reference cast to pointer to the same complete array | 6 | 0 | pass |
| Explicit array-element borrow alias control | 2 | 0, original unchanged after write | refusal |

The last row exposes an independent element-borrow alias defect. It is preserved, not fixed by the library change or claimed safe by the cast proposal. The status of the diagnostic collector is not the status of every behavior.

## Slurm 19480

Root's existing C++23 ptrace tracer and its source are copied without modification. It observes signals rather than installing breakpoints. The six producers run in both modes. The three value-cast cases fault; ref4, heap_f64 and same_array_mut_write exit zero. For value4, the actual fault reports RDI=1, RSI=0, fault address=1 and SIGSEGV. This directly corroborates the managed handle being used as a raw address. Reading a slot through ref4 does not signal, but returns the wrong bits; lack of a signal is not correctness.

## Proposed checker boundary remains an artifact

The unadopted E398 proposal exists only under /workspace/.tmp/array-cast-guard12/input. It targets both checker cast spines, refusing array VALUE to raw pointers and references to arrays cast to pointers of another pointee type, extent, element type or increased mutability. References to the same complete array and existing raw heap operations must continue to work. No lower.sio, read_f64 or native codegen edit is part of it.

Nested generic/unknown type equality and the imported/BFS impact are not established by the syntactic inventory. The owner of the independent ref-to-raw regression must review the boundary before any adoption. No checker proposal is included in the library commit.
