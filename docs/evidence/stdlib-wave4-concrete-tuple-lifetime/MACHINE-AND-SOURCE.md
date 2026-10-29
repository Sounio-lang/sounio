# Observation, machine patterns and source ownership

Observed fact: only the nested concrete values change after a real Sounio call,
on both legacy artifacts; flat remains stable. This by itself is interference,
not a general stack-memory proof.

Actual generated legacy code adds stronger evidence. The delimited first function
window starts0x401000 and has the standard RBP prologue, savedR12 and112-byte frame.
Its inner tuple buffer address is produced by LEA RAX,[RBP-72] at0x401083. That
address is stored as a cell in the outer buffer. The outer buffer address is
LEA RAX,[RBP-104] at0x4010bb. At0x4010c5 the code copies exactly two qwords from the
outer buffer to the caller-provided R12 SRet destination and returns that pointer.
The inner address is copied as a word; the child buffer remains in the callee frame.

The clobber function starts0x4010f5, has a4144-byte RBP frame, and initializes the
array with LEA RDI,[RBP-4104]; MOV ECX,512; REP STOSQ at0x40111f. This exact machine
sequence, within the known prologue/body, establishes an actual RBP-relative
512-word stack fill in these legacy ELFs. It is not inferred merely from an array
declaration. Callsite byte windows show calls to these functions and the ordinary
outgoing argument cleanups. The two compiler artifacts emit identical witness ELFs.

The pattern scanner is deliberately NOT a general x86 disassembler. It reports
known byte sequences and candidate frames; its unrelated matches elsewhere in the
included builtin code are not assigned source meanings. Byte windows are extracted
directly from the executable ELF load mapping by the independent C++23 reader.
No runtime RSP/pointer trace or Lean lifetime proof is claimed.

Qualified ccfa producer source SHA16cab9c8700981608469f052a96733360e822db6598221432f438d404cd2913b:
- ty_slot_count's Ty6 branch sizes registered structs; an unregistered synthetic
  tuple falls through to one cell.
- Tuple literal emission saves each element word, allocates the outer buffer and
  copies an aggregate only when its recorded element slot count exceeds one.
- In this witness the nested child is represented by one pointer cell.
- stabilize_return_agg_x86 copies only the top-level nslots to the caller SRet
  destination; the two-word machine copy matches that path.

This combination leaves a child-frame address reachable through a caller-owned
outer return buffer. The interpretation is a legacy nested-return lifetime defect,
supported by concrete before/after values plus this source/machine path. A pointer
cell representation is not itself declared wrong: how child storage must be owned,
copied or promoted is an architectural decision deferred to root.

Raw04's actual sealed source lower_tuple_literal_expr_ref emits IrAlloc and stores
its element values into the aggregate; its native output passes the same cases in
both loop modes. Its scanner does not find the legacy RBP fill512/SRet copy patterns.
Absence of a byte pattern is not proof of a particular allocation backend; no blanket
claim that every modern allocation is heap/mmap is made.

The C++23 reference uses its OWN noinline volatile512-cell stack buffer to compute
the exact checksum and expected values. It never clobbers the Sounio process or
proves Sounio frame layout across processes.
