# Full-IR prerequisite and must-definition acceptance

Final acceptance requires complete stage-bound schema2 source capture and a CFG
must-definition audit for EVERY real function, not only names wide_counts/core.

The existing strict parser is tests/fixtures/wave4_loop_checkpoint_elision/
nv2_trace_full_reference.cpp, CLI --check source COMPLETE_COMPILE.stdout.
Its success proves diagnostic structure/census, not register dominance.

The earlier collector at compiler-resumed/wide-mustdef01/reference (job19724),
source SHA8357c5fff73c2bb58888c8112a0e9d1bd60e515dce9197ae762101d4ec5cae84,
returns0 even when it prints CFG_UNDEFINED. Never use process status alone.
Its current target-selection completeness still needs review before reuse as
a global acceptance auditor. Parent/text lane was informed of this distinction.

Required auditor contract before connecting it to this harness:

- Cross-check the parser function count, indices, names and executable/data/stub
  census; audit all executable functions or report explicit unsupported coverage.
- Treat real parameter registers as entry definitions, only actual CFG predecessors
  as incoming paths, and join by intersection. Iterate loops to convergence.
- Reject undefined reachable operands and zero-valued registers absent on a path.
  Never repair the IR by assigning missing registers zero.
- Require total_missing=0 with nonempty executable coverage and exact census.
  A collector that filters only core names cannot close the package.
- Preserve before19737 missing3, candidate zero missing and source/ELF identity.
  Original wrong-assert native control must continue to fail.
- Audit LOOP_REGIONS0/1, default/-O and WIDE literal/dyadic/hoist captures; a FULL
  formatter refusal is an explicit blocked diagnostic, not an empty accepted arena.

Pending until an actual root-authorized compiler snapshot contains the reviewed
patch. This document does not claim the proposed transformation has passed.
