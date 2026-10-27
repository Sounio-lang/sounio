# Integrated acceptance controls

Slurm 18712 COMPLETED 0:0: runner controls 0, reliability controls 0 (12 cases), source guard 0. Snapshot: /scratch/sounio-stdlib-wave3-20261027/acceptance-sprint-20261027/src. Receipt logs below are copied from that execution.

The full-suite runner rejects all-skip results. --acceptance requires --slow and rejects any skipped test; ordinary diagnostic runs retain mixed pass/skip reporting. Reliability aggregation rejects zero executed passes, non-pass producer status and selection errors. Controls cover all-skip, mixed diagnostic/acceptance, fully passing acceptance and missing cases.

Slurm 18711 failed because PATH lacked the directory containing node for child processes. The identical source snapshot passed in 18712 after PATH correction. Both receipts are preserved. These are harness controls, not execution of the complete stdlib corpus.

The integrated deep gate now requires mechanics and phonon Quantity alongside the existing science phases; CI preserves their artifacts. These newly composed phases still require a complete integrated run.

Final compatibility closure: Slurm 18718 COMPLETED 0:0 against the compat/src snapshot, runner controls 0, reliability controls 0 (13 cases), source guard 0. Legacy v1 receipts omit status_summary; they remain valid when counts, rows and producer exits reconcile. A present non-pass status and selection errors cannot be promoted to pass. New compatibility control covers the unchanged legacy producer contract. compat/ contains the final receipts.
