# Reuse BigFrame row scratch storage

bf_unique and bf_drop_duplicates now allocate their64-element row scratch array once per operation and overwrite its used columns before each bf_push_row. The output order, hash probing, column count and growth behavior are unchanged.

Slurm19159,24GiB allocation, compares the same raw compiler and identical4million-row input against the old and changed library. SOUNIO_LOOP_REGIONS=0 in both: old library checks/compiles but native exits181; changed library checks/compiles and native exits0 with BIGFRAME_ROW_REUSE_OK. Input/compiler/ELF guards pass. Earlier19143/19147 runs hit scheduler OOM with6GiB while compiling and do not qualify either library.

The portable gate in19174 strengthens the postcondition by checking every unique key and every deduplicated key/payload across all4million output rows. Native exit0, exact marker and all hash guards pass. Reproduce on a scheduled Linux x86-64 worker with sufficient compiler memory (24GiB qualified):

    MADAROS_RAW_BIN=/absolute/rebuilt/madaros bash scripts/stdlib/bigframe_row_reuse_gate.sh /absolute/fresh/output

The original test_bigframe_ops_stdlib.sio remains unchanged and blocked by the separate borrow binding-capacity defect. This focused receipt does not close that original end-to-end gate or full stdlib acceptance.
