# Typed local receiver for graph slice length

ProbabilityMeasure.support_size binds its existing nodes slice to a local explicitly typed[NodeId], then calls nodes.len(). Public signatures, slice data, mutation behavior and return type remain unchanged. This lets the current lowerer select the existing slice-length intrinsic; no compiler/lowerer file is changed. The general direct-field receiver lowering defect remains separate.

Slurm19600 completed0 in8s with raw8abe and the archived raw02 SDK. In both loop-region modes, the original Graph program checks0 but fails compilation1 before the change: native emission refuses empty stub len with missing_lowered_body. The before refusal and source are preserved. With only this receiver change, that same original program checks0/compiles0/executes0 and prints GRAPH_TYPES_OK. It checks the four-edge undirected graph, degree and edge counts, neighbor contents, absent edge, and visibility of mutations through shared Seq handles. No original assertion or test file was changed.

A separate program calls the actual public ProbabilityMeasure API using empty, one-element and four-element node slices. Both modes compile and execute0; C++23 requires exactly SIZE0=0, SIZE1=1 and SIZE4=4, with no missing/extra rows. These checks exercise actual lengths and cannot be satisfied by a constant-return stub. All original source, input, raw and executable final verification manifests passed. This is focused API qualification; the full graph module family and whole stdlib remain open.

The exact before/after library source is retained in tests/fixtures/wave4_graph_slice. Complete snapshots and native artifacts are retained separately; durable custody will be linked after readback.
