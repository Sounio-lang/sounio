Final component is19800/revision02; initial19797 coverage below is historical. See REVISION-02.md and SEMANTICS.md.

# Independent C++23 global CFG must-definition component

The auditor consumes a COMPLETE schema2 source capture. It reuses the existing
strict nv2_trace_full_reference.cpp parser unchanged, then reconstructs every
function/instruction from that parser's validated rows and checks the exact
function/instruction census again. It never invokes Sounio, applies the proposed
shadow patch, changes IR, or substitutes zero for a missing definition.

CLI: reference --audit COMPLETE_COMPILE.stdout

Exit0 requires a nonempty executable corpus, complete supported coverage and
total_missing=0. Missing definitions return1. Unsupported operand/CFG/ABI semantics
return2. Malformed structure, bounds or edges return3. A summary with coverage=complete
is produced only after every function has been accounted for.

All executable functions are audited, regardless of name. Empty descriptors and
DATA descriptors are enumerated separately; EXTERN_EMPTY_DESCRIPTOR describes the
captured empty IR descriptor and does not prove a source declaration was external.
The strict parser validates their empty/DATA shape. No function is silently skipped.

Operand semantics0..22 follow the declaration and constructors in self-hosted/ir/
ir.sio. In particular, IrNop float markers do NOT define a value; calls read the
complete captured argument list; IndexSet also reads the register encoded in its
integer immediate. Phi and every opcode above22 are explicit coverage refusals
because no certified incoming-edge semantics has been implemented. f128 parameter
spans and reachable implicit function fallthrough also refuse coverage. This is
a deliberately scoped auditor, not full verification of all backend opcodes.

Actual param_regs are entry definitions; optional sret destination is also an
entry definition when its metadata is valid. Virtual register count uses dynamic
sets up to the official16384 limit, with a tested1025-register control. Labels and
branch targets are validated, duplicate labels are refused, and reachability is
computed from instruction0. Incoming must-def sets intersect all reachable
predecessors and iterate loops from top to a fixed point. Uses are checked BEFORE
the same instruction's definition, preserving same-block ordering and loopback
constraints. Unreachable instructions still require supported opcode/shape;
their unreachable uses do not become false missing-definition findings.

Slurm19797 COMPLETED0/4s/206244K:
-21 synthetic full-schema controls passed.
- Original immutable WIDE19724 capture:19 descriptors,3 executable functions,
  16 empty; both cores have3 missing uses, total6. Audit status1.
- Original scalar19737 capture:14 descriptors,2 executable functions,12 empty;
  core has3 missing uses, total3. Audit status1.
- Source, capture, CXX and auditor identity guards all passed.

The component gate passes because it correctly REJECTS these defective baselines.
This is not qualification of a new compiler, the pending variance-shadow patch or
a repaired WIDE pipeline. No new source capture was requested.

19796 failed before C++ build because the private tools environment required R.
Its stderr/recipe are retained.19797 defines R explicitly and is the actual
component result. Recipes are identified session copies, not Slurm spool readback.

Current captured source/SDK/compiler graph admission remains a separate root
responsibility. A structurally consistent but maliciously resealed capture with
a function removed cannot be recognized from that capture alone: bind expected
source graph and capture byte identity externally. The drop-function control
demonstrates refusal when the original complete census is retained.
