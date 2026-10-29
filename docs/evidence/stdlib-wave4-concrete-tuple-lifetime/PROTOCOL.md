# Reproduction and provenance

MCP sounio_verificado was called first for the new small concrete fixture. Its one
candidate failed compilation and had mismatched return shape/unresolved variables.
It was not adopted or counted as qualification. The corrected sources here were
then compiled natively under Slurm.

Sounio compilation deadline120s, native deadline30s, stack512MiB, one CPU/8GiB.
C++23 references and binary inspection are also built/run only under Slurm.
No numerical Python/Rust/Julia or new FFI/runtime operation is used.

The same immutable source files feed9cf,ccfa andraw04. Raw04 uses the qualified
R05 stdlib snapshot and LOOP_REGIONS0/1 with checkpoint elision disabled. The legacy
source outputs include their own integrated parser/typecheck diagnostics. Explicit
diagnostic errors reject native admission even when compiler exit is0.

Each ELF has magic checked before execution and SHA before/after native. Compiler
artifacts, source, CXX/Node/helper inputs and completev2 SDK stdlib identities are
guarded before/after. All compiler/native/oracle/machine statuses remain separate;
the outer audit execution status0 is NOT aggregate numerical acceptance.

Compiler identities:
9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a
ccfaec48d1bae176091ef1f3401c2d863c1f05284390f3c50ca9d66e94c7114f
raw04 a4f5e8d655754202493bbb04d8fbb15890b10e8fc861ac3dc60438452704c333

The ccfa source is the actual root05 private producer copy SHA16cab..., checked
against original legacy-flow-bounds03/candidate.sio. Exact source identity for the
old9cf emitter is not independently asserted here; its ELF and emitted witnesses
are pinned. Raw04 source excerpts come from its actual sealed source snapshot,
not a newer working lower.sio.

19847 COMPLETED0/1s/83372K performs only immutable ELF equality/byte-window reading
with C++23, including an invalid-address refusal. It does not rerun Sounio oracles.
All windows and original statuses are retained. Recipes are session copies, not
original Slurm spool files. Original19818 sources/results are referenced unchanged.

This independent result does not prove that every detail of the Option-bearing
private19818 failure is identical. That original previously failed compilation on
ccfa, so its full before/after numerical route is different. The concrete witness
establishes an existing legacy nested-return lifetime failure without requiring
that inference candidate; original19818 expectations and artifacts stay separate.
