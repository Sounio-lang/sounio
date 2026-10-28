# DP54 event root scratch ownership repair

Whole integrated replay19392 failed dp54_invalid_probe during checking with E038 in solve_dp54_event64: the call borrowed &w.trial and &!w.k6 from the same aggregate. Keep the accepted trial shared and reuse a separate [f64;64] root-query scratch allocated once before the solve loop. No equations, integration limits, tolerances, event contracts, APIs, or assertions changed.

Slurm19472 COMPLETED0 ran the existing dp54_dense_events_gate.sh against raw11332ed8a6cf7ccb85f0a31bf63bed1fcc10659054d68639d6dba233a6014ff4 with SOUNIO_LOOP_REGIONS=1. Six original programs checked/compiled/ran0; the analytic, invalid-input, and baseline streams agreed with the independent C++23 oracle. All nine corrupt-stream controls were refused1, as were dense/event private access and repeated linear consumption. Executable, source, compiler, generated-source, Boost header, and GCC final guards passed. This is a focused gate, not a new whole-corpus result.

Initial staging19466 failed before compilation because bin/madaros and the referenced callback fixture were absent.19468 reached the independent C++ reference but failed because the compiler lacked the Boost include path.19472 uses the identified existing Boost18302 header SDK through CPATH; neither reference nor oracle tolerance was changed. Both earlier failures remain retained.
