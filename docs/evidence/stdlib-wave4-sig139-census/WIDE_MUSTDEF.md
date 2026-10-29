# Scientific-loop must-definition diagnostic, separate from legacy return failure

This is an independent read-only diagnostic of the complete standalone WIDE
projection captured by root/core19709/19667. It is not imported QPU acceptance,
a numerical rerun, an arena reconstruction or an elision qualification.

Job19724 completed0 in3seconds. Its C++23 program first requires a complete
Schema2 source-stage capture with the existing strict parser (19functions,
634instructions,37call arguments). It then computes reachable CFG predecessors
and standard must-defined register sets, independently of the Sounio pass's
bitmap implementation. The two core functions converge in2iterations.

Both wide_core_literal and wide_core_hoist have three missing definitions:
reg16 is used at instruction82(BinOp),92(BinOp),97(Copy). main has none.
The source trace shows reg16=integer-zero bits plus a floating marker at17/18,
inside the conditional seed==0 path. BranchFalse15 targetsLabel0 at22 and
bypasses those definitions when the seed is nonzero. Later floating/shadow
arithmetic reuses reg16 in the loop. The input metadata/register/helper profile
is otherwise not the previously suspected ref/SRet admission issue.

This corroborates the actual per-function LCE status10 captured independently
by core19709. The pass correctly refuses the missing must-definition proof.
No weakening of the pass, mathematical source rewrite, favorable seed selection
or change to the scientific oracle is proposed. The captured source-stage IR is after optional opt_cleanup_module_inplace.
This establishes the missing definition at that stage only: lower construction,
CSE or cleanup may be responsible. No source cache is diagnosed here. The exact
compiler decision remains open until a same-source/same-raw comparison without
cleanup or before it; lower.sio remains untouched.

The graph auditor is a diagnostic reader over a complete capture. It does not
mint compiler authority, construct IrModule/arena objects, mutate the pass, or
replace the actual analyzer's result. INPUTS.sha256 binds the complete trace,
C++23 executable and parser sources; audit.stdout records the exact sites.
Original capture path:
 /scratch/sounio-stdlib-wave3-20261027/root-integrated04-capture/results/wide_projection.1.compile.stdout

The legacy implicit-return failure is separate: Packet cases19720 fail only in
seed9cf while the modern compiler passes them. This scientific-loop observation
concerns a modern-produced IR operand and does not establish the same cause.

The combined supplement archive was sealed/read back by19726:
 /orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/sig139-census01/packet-oracle-and-wide-cfg-supplement.tar.gz
 SHA25696ce75947c9c4fa985f4377244ce6bbfd92a43030461f547e5554c549142e98a.
It preserves the complete input trace, parser/auditor source, C++23 executable,
logs and the independent Packet field oracle. No mathematical source or pass
was changed to obtain the observed statuses.
