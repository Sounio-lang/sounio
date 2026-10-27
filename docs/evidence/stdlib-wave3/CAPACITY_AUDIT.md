# Capacity source audit in Node

The IR and MIR coherence gates now run source metadata inspection in Node.
They preserve strict unique-declaration matching, expected capacity checks,
analysis refusal checks, serialization bounds, detectable refusal counters,
and MIR type/initializer agreement. No Python implementation was executed.

The initial port attempt (Slurm 18135) failed because the old gate still
expected an inline IrModule function array. The current a955 source uses
IrFunctionTable with bounded heap chunks. The revised source audit binds the
module handle, chunk entry count, table index count, their product against
IR_MAX_FUNCS, both index guards and refusal latches, division/index arithmetic,
chunk initialization, and the serializer's bounded pool insertion path.
It continues to reject reintroduced inline function serialization.

Slurm 18139 passed the IR and MIR gates and 25 positive/refusal controls.
Mutants remove or duplicate capacity declarations, restore the obsolete inline
instruction field, remove the region handle, disable DCE/CP guards, break
function-pool dimensions/divisors/index guards/chunk initialization, remove
serializer pool insertion, change string-table dimensions, remove detectable
refusals, or break MIR field/initializer/refusal agreement.

This is source metadata evidence. It does not establish runtime readback,
optimizer correctness, allocation-layout safety, or a working larger limit.
The nonoptional opt_cleanup register peels remain a separate native gate;
the real knowledge_octonion_structure before/after optimizer behavior must
pass before a capacity increase is accepted.
